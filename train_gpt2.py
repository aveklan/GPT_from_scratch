import os
import math
import time
import inspect
from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from tensor_parallel import (
    TPLinear, TPLinearReduce, VocabParallelEmbedding, shard_tensor, partition_sizes,
    vocab_parallel_cross_entropy, gather_logits, clip_grad_norm_tp,
    full_state_dict, load_full_state_dict,
)


class CausalSelfAttention(nn.Module):

    def __init__(self, config, tp_rank=0, tp_world_size=1):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        local_heads = partition_sizes(config.n_head, tp_world_size)[tp_rank]
        head_size = config.n_embd // config.n_head
        # Column-parallel QKV projection
        self.c_attn = TPLinear(config.n_embd, 3 * config.n_embd, tp_rank=tp_rank, tp_world_size=tp_world_size, stride=3, partition_unit=head_size)
        # Row-parallel output projection
        self.c_proj = TPLinearReduce(config.n_embd, config.n_embd, tp_rank=tp_rank, tp_world_size=tp_world_size, partition_unit=head_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = local_heads
        self.n_embd = local_heads * head_size
        self.head_size = config.n_embd // config.n_head

    def forward(self, x):
        B, T, _ = x.size()
        # Q, K and V for local heads only; no intermediate communication.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        # Output projection with AllReduce
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config, tp_rank=0, tp_world_size=1):
        super().__init__()
        # Column-parallel expansion
        self.c_fc = TPLinear(config.n_embd, 4 * config.n_embd, tp_rank=tp_rank, tp_world_size=tp_world_size)
        self.gelu = nn.GELU(approximate="tanh")
        # Row-parallel contraction
        self.c_proj = TPLinearReduce(4 * config.n_embd, config.n_embd, tp_rank=tp_rank, tp_world_size=tp_world_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config, tp_rank=0, tp_world_size=1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, tp_rank=tp_rank, tp_world_size=tp_world_size)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config, tp_rank=tp_rank, tp_world_size=tp_world_size)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class GPT(nn.Module):

    def __init__(self, config, tp_rank=0, tp_world_size=1):
        super().__init__()
        self.config = config
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        if tp_world_size < 1 or not 0 <= tp_rank < tp_world_size:
            raise ValueError("Invalid tensor-parallel rank/world size")
        if tp_world_size > 1:
            if not dist.is_initialized() or dist.get_world_size() != tp_world_size or dist.get_rank() != tp_rank:
                raise ValueError("The default process group must match the TP rank/world size")

        self.transformer = nn.ModuleDict(
            dict(
                wte=VocabParallelEmbedding(config.vocab_size, config.n_embd, tp_rank, tp_world_size),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config, tp_rank=tp_rank, tp_world_size=tp_world_size) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        padded_vocab = self.transformer.wte.full_shape[0]
        self.lm_head = TPLinear(config.n_embd, padded_vocab, bias=False,
                                tp_rank=tp_rank, tp_world_size=tp_world_size)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (TPLinear, TPLinearReduce, VocabParallelEmbedding)):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            # Same seed on all ranks: initialize one full CPU tensor at a time,
            # then retain this rank's slice. Replicas stay identical, shards differ.
            full = torch.empty(module.full_shape, dtype=module.weight.dtype)
            torch.nn.init.normal_(full, mean=0.0, std=std)
            with torch.no_grad():
                module.weight.copy_(shard_tensor(full, module.weight.tp_dim,
                    self.tp_rank, self.tp_world_size, module.stride, module.weight.tp_unit))
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, gather_output=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # local vocabulary partition
        start = self.tp_rank * logits.size(-1)
        valid = torch.arange(logits.size(-1), device=logits.device) + start < self.config.vocab_size
        logits = logits.masked_fill(~valid, float("-inf"))
        loss = None
        if targets is not None:
            loss = vocab_parallel_cross_entropy(
                logits, targets, self.tp_rank, self.tp_world_size).mean()
        # Training uses the distributed loss without gathering large logits.
        # Inference/evaluation callers get conventional full logits by default.
        if gather_output is None:
            gather_output = targets is None
        if gather_output:
            logits = gather_logits(logits, self.tp_rank, self.tp_world_size, self.config.vocab_size)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, tp_rank=0, tp_world_size=1):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        if tp_world_size > 1:
            parallel_model = cls(config, tp_rank, tp_world_size)
            load_full_state_dict(parallel_model, model.state_dict())
            return parallel_model
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.tp_rank == 0:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if self.tp_rank == 0:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


# -----------------------------------------------------------------------------
import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root="edu_fineweb10B"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# ============================================================================
# Tensor Parallelism Training
# Usage: torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 train_gpt2.py
# ============================================================================

from torch.distributed import init_process_group, destroy_process_group

def main():
    import argparse
    import tiktoken
    from contextlib import nullcontext

    parser = argparse.ArgumentParser(description="GPT-2 with Megatron tensor parallelism")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--total-batch-size", type=int, default=2048, help="Global tokens per optimizer step")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=19073)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--data-root", default="edu_fineweb10B")
    parser.add_argument("--log-dir", default="log")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--skip-eval", action="store_true", help="Skip validation, HellaSwag and generation for smoke runs")
    args = parser.parse_args()
    if min(args.batch_size, args.total_batch_size, args.sequence_length, args.max_steps) < 1:
        parser.error("Batch sizes, sequence length and max steps must be positive")

    # One default process group contains all shards of one model.
    tp = int(os.environ.get("RANK", -1)) != -1  # is this a tp run?
    if tp:
        use_cuda = args.device != "cpu" and torch.cuda.is_available()
        if args.device == "cuda" and not use_cuda:
            raise RuntimeError("CUDA requested but unavailable")
        tp_rank = int(os.environ["RANK"])
        tp_local_rank = int(os.environ["LOCAL_RANK"])
        tp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{tp_local_rank}" if use_cuda else "cpu"
        if use_cuda:
            torch.cuda.set_device(device)
        init_process_group(backend="nccl" if use_cuda else "gloo")
        master_process = tp_rank == 0  # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-TP run
        tp_rank = 0
        tp_local_rank = 0
        tp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if args.device != "cpu" and torch.cuda.is_available():
            device = "cuda"
        elif args.device == "cuda":
            raise RuntimeError("CUDA requested but unavailable")
        print(f"using device: {device}")

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    if args.dtype == "bfloat16" and device_type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This GPU does not support native BF16; use --dtype float32")

    def autocast_context():
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16) if args.dtype == "bfloat16" else nullcontext()

    if not args.skip_eval:
        from hellaswag import render_example, iterate_examples
        enc = tiktoken.get_encoding("gpt2")

    total_batch_size = args.total_batch_size
    B = args.batch_size  # shared microbatch, not per-rank distinct data
    T = args.sequence_length
    assert (
        total_batch_size % (B * T) == 0
    ), "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        print(f"Tensor Parallel World Size: {tp_world_size}, Rank: {tp_rank}")

    train_loader = None if not master_process else DataLoaderLite(
        B=B, T=T, process_rank=0, num_processes=1, split="train", data_root=args.data_root
    )
    val_loader = None if args.skip_eval or not master_process else DataLoaderLite(
        B=B, T=T, process_rank=0, num_processes=1, split="val", data_root=args.data_root
    )

    def next_shared_batch(loader):
        if master_process:
            x, y = loader.next_batch()
            batch = torch.stack((x, y)).to(device)
        else:
            batch = torch.empty((2, B, T), dtype=torch.long, device=device)
        if tp:
            dist.broadcast(batch, src=0)
        return batch[0], batch[1]

    torch.set_float32_matmul_precision("high")

    # create model WITH TENSOR PARALLELISM PARAMETERS
    model = GPT(GPTConfig(vocab_size=args.vocab_size, block_size=T, n_layer=args.n_layer,
                          n_head=args.n_head, n_embd=args.n_embd), tp_rank=tp_rank, tp_world_size=tp_world_size)
    # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
    model.to(device)
    use_compile = (
        False  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    )
    if use_compile:
        model = torch.compile(model)
    raw_model = model  # no DDP wrapper needed for TP

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 0
    max_steps = args.max_steps


    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)


    # optimize!
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device_type=device_type
    )

    # create the log directory we will write checkpoints to and log to
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    if master_process:
        with open(log_file, "w"):
            pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        # once in a while evaluate our validation loss
        if not args.skip_eval and (step % 250 == 0 or last_step):
            model.eval()
            if master_process:
                val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = next_shared_batch(val_loader)
                    with autocast_context():
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        # once in a while evaluate hellaswag
        if (step % 250 == 0 or last_step) and (not use_compile) and not args.skip_eval:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # Every TP rank must process the same example and collectives.
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with autocast_context():
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile) and not args.skip_eval:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with autocast_context():
                        logits, loss = model(xgen)  # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    # get the probabilities
                    # The padded GPT vocabulary includes IDs the tokenizer cannot decode.
                    logits = logits[:, :enc.n_vocab]
                    probs = F.softmax(logits.float(), dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    if tp:
                        dist.broadcast(xcol, src=0)
                    # append the same sampled token on every rank
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            if master_process:
                for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = enc.decode(tokens)
                    print(f"rank {tp_rank} sample {i}: {decoded}")

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = next_shared_batch(train_loader)
            with autocast_context():
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        norm = clip_grad_norm_tp(raw_model, 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = (
            B * T * grad_accum_steps
        )
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

        if (step > 0 and step % 5000 == 0) or last_step:
            # All ranks participate; only rank zero writes the reconstructed model.
            state = full_state_dict(raw_model)
            if master_process:
                torch.save({"model": state, "config": asdict(raw_model.config), "step": step},
                           os.path.join(log_dir, f"model_{step:05d}.pt"))
            del state

    if tp:
        destroy_process_group()

if __name__ == "__main__":
    main()
