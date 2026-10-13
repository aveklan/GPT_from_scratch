"""Megatron (arXiv:1909.08053, section 3) using native PyTorch only.

The default process group is one tensor-parallel model; inputs and replicated
parameters must be identical on all ranks. No data or sequence parallelism.
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class _Copy(torch.autograd.Function):
    """Megatron f: identity forward, SUM all-reduce backward."""

    @staticmethod
    def forward(ctx, x, world_size):
        ctx.world_size = world_size
        return x

    @staticmethod
    def backward(ctx, grad):
        if ctx.world_size > 1:
            grad = grad.contiguous().clone()
            dist.all_reduce(grad)
        return grad, None


class _Reduce(torch.autograd.Function):
    """Megatron g: SUM all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, x, world_size):
        if world_size > 1:
            x = x.contiguous().clone()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _Gather(torch.autograd.Function):
    """Gather vocabulary logits for inference; backward selects the local slice."""

    @staticmethod
    def forward(ctx, x, rank, world_size):
        ctx.rank, ctx.world_size = rank, world_size
        if world_size == 1:
            return x
        parts = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        return grad.chunk(ctx.world_size, dim=-1)[ctx.rank].contiguous(), None, None


def shard_tensor(tensor, dim, rank, world_size, stride=1):
    """Stride 3 partitions Q, K and V separately, preserving local [Q, K, V]."""
    if dim is None:
        return tensor
    return torch.cat(
        [part.chunk(world_size, dim=dim)[rank]
         for part in tensor.chunk(stride, dim=dim)], dim=dim
    ).contiguous()


def mark_sharded(parameter, dim, stride=1):
    parameter.tp_dim = dim
    parameter.tp_stride = stride


class TPLinear(nn.Module):
    """Column parallel: shard output features; keep its output local."""

    def __init__(self, in_features, out_features, bias=True, tp_rank=0,
                 tp_world_size=1, stride=1):
        super().__init__()
        if out_features % (tp_world_size * stride):
            raise ValueError("Output features must divide evenly across TP ranks and QKV groups")
        self.tp_rank, self.tp_world_size = tp_rank, tp_world_size
        self.full_shape = (out_features, in_features)
        self.stride = stride
        self.weight = nn.Parameter(torch.empty(out_features // tp_world_size, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features // tp_world_size)) if bias else None
        mark_sharded(self.weight, 0, stride)
        if bias:
            mark_sharded(self.bias, 0, stride)

    def forward(self, x):
        return F.linear(_Copy.apply(x, self.tp_world_size), self.weight, self.bias)


class TPLinearReduce(nn.Module):
    """Row parallel: consume local input features, sum outputs, then add bias."""

    def __init__(self, in_features, out_features, bias=True, tp_rank=0, tp_world_size=1):
        super().__init__()
        if in_features % tp_world_size:
            raise ValueError("Input features must divide evenly across TP ranks")
        self.tp_rank, self.tp_world_size = tp_rank, tp_world_size
        self.full_shape = (out_features, in_features)
        self.stride = 1
        self.weight = nn.Parameter(torch.empty(out_features, in_features // tp_world_size))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        mark_sharded(self.weight, 1)

    def forward(self, x):
        out = _Reduce.apply(F.linear(x, self.weight), self.tp_world_size)
        return out + self.bias if self.bias is not None else out


class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd, tp_rank=0, tp_world_size=1):
        super().__init__()
        self.tp_rank, self.tp_world_size = tp_rank, tp_world_size
        self.local_size = (vocab_size + tp_world_size - 1) // tp_world_size
        self.start = tp_rank * self.local_size
        self.full_shape = (self.local_size * tp_world_size, n_embd)
        self.stride = 1
        self.weight = nn.Parameter(torch.empty(self.local_size, n_embd))
        mark_sharded(self.weight, 0)

    def forward(self, tokens):
        outside = (tokens < self.start) | (tokens >= self.start + self.local_size)
        local_tokens = (tokens - self.start).masked_fill(outside, 0)
        out = F.embedding(local_tokens, self.weight).masked_fill(outside.unsqueeze(-1), 0)
        return _Reduce.apply(out, self.tp_world_size)


class _VocabCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, rank, world_size):
        # FP32 softmax/normalization for low precision, retaining FP64 for checks.
        scores = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits.clone()
        maximum = scores.max(dim=-1).values
        if world_size > 1:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        scores = scores - maximum.unsqueeze(-1)
        local_size = scores.size(-1)
        local_targets = targets - rank * local_size
        outside = (local_targets < 0) | (local_targets >= local_size)
        local_targets = local_targets.masked_fill(outside, 0)
        selected = scores.gather(-1, local_targets.unsqueeze(-1)).squeeze(-1)
        selected = selected.masked_fill(outside, 0)
        exp = scores.exp()
        denominator = exp.sum(dim=-1)
        if world_size > 1:
            dist.all_reduce(selected)
            dist.all_reduce(denominator)
        probabilities = exp / denominator.unsqueeze(-1)
        ctx.save_for_backward(probabilities, local_targets, outside)
        ctx.input_dtype = logits.dtype
        return denominator.log() - selected

    @staticmethod
    def backward(ctx, grad):
        probabilities, targets, outside = ctx.saved_tensors
        result = probabilities.clone()
        result.scatter_add_(-1, targets.unsqueeze(-1),
                            -((~outside).to(result.dtype)).unsqueeze(-1))
        return (result * grad.unsqueeze(-1)).to(ctx.input_dtype), None, None, None


def vocab_parallel_cross_entropy(logits, targets, rank, world_size):
    return _VocabCrossEntropy.apply(logits, targets, rank, world_size)


def gather_logits(logits, rank, world_size, vocab_size):
    return _Gather.apply(logits, rank, world_size)[..., :vocab_size]


@torch.no_grad()
def clip_grad_norm_tp(model, max_norm):
    """Global L2 norm, counting replicated parameters and tied weights once."""
    params = [p for p in model.parameters() if p.grad is not None]
    norm_squared = torch.zeros((), device=params[0].device, dtype=torch.float64)
    for p in params:
        if getattr(p, "tp_dim", None) is not None or model.tp_rank == 0:
            norm_dtype = torch.float64 if p.grad.dtype == torch.float64 else torch.float32
            norm_squared += torch.linalg.vector_norm(p.grad, dtype=norm_dtype).double().square()
    if model.tp_world_size > 1:
        dist.all_reduce(norm_squared)
    norm = norm_squared.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for p in params:
        p.grad.mul_(scale.to(p.grad.dtype))
    return norm


@torch.no_grad()
def full_state_dict(model):
    """Collectively reconstruct a portable CPU state dict, returned on rank 0 only."""
    result = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        value = p.detach()
        dim = getattr(p, "tp_dim", None)
        if dim is not None and model.tp_world_size > 1:
            parts = [torch.empty_like(value) for _ in range(model.tp_world_size)]
            dist.all_gather(parts, value.contiguous())
            stride = p.tp_stride
            split = [part.chunk(stride, dim=dim) for part in parts]
            value = torch.cat([split[r][s] for s in range(stride)
                               for r in range(model.tp_world_size)], dim=dim)
        if model.tp_rank == 0:
            if name in ("transformer.wte.weight", "lm_head.weight"):
                value = value[:model.config.vocab_size]
            result[name] = value.cpu().clone()
    return result if model.tp_rank == 0 else None


@torch.no_grad()
def load_full_state_dict(model, state):
    """Load a conventional GPT state dict, slicing the weights for this rank."""
    expected = dict(model.named_parameters(remove_duplicate=False))
    if expected.keys() != state.keys():
        raise ValueError("Checkpoint keys do not match the model")
    if not torch.equal(state["transformer.wte.weight"], state["lm_head.weight"]):
        raise ValueError("Checkpoint embedding and output weights must be tied")
    for name, p in expected.items():
        value = state[name]
        if name in ("transformer.wte.weight", "lm_head.weight"):
            padding = p.size(0) * model.tp_world_size - value.size(0)
            if padding > 0:
                value = F.pad(value, (0, 0, 0, padding))
        dim = getattr(p, "tp_dim", None)
        value = shard_tensor(value, dim, model.tp_rank, model.tp_world_size,
                             getattr(p, "tp_stride", 1))
        if value.shape != p.shape:
            raise ValueError(f"Shape mismatch for {name}: {value.shape} != {p.shape}")
        p.copy_(value)
