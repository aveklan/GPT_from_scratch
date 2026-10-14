# GPT-2 tensor parallelism

This implements the tensor-parallel algorithm from section 3 and Figure 3 of
[Megatron-LM (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053), using
PyTorch distributed collectives and autograd. No additional TP library is needed.
The nanoGPT architecture and this branch's 12-layer / 12-head / 768-width defaults
are retained; this does not reproduce the paper's entire training recipe.

## Partitioning and communication

- QKV is column parallel. Each rank holds matching slices of Q, K, and V and
  computes only its local attention heads. The fused weight layout is
  `[Q_local, K_local, V_local]`.
- The first MLP projection is column parallel; GELU runs on local features.
- Both output projections are row parallel, consuming local features. Their
  partial outputs are summed before adding the replicated bias.
- Megatron's `f` is identity forward / sum backward; `g` is sum forward /
  identity backward. Each block has two forward and two backward all-reduces.
  There are no attention/MLP intermediate all-gathers.
- Input embeddings and the tied output projection are partitioned by vocabulary.
  Input embeddings are summed; distributed cross-entropy reduces token-level
  maxima, target scores, and exponential sums instead of gathering full logits.
  Vocabulary sizes not divisible by TP size are padded internally; extra entries
  are excluded from softmax and removed when gathering logits or checkpoints.
- LayerNorm, position embeddings, residuals, and row-projection biases are
  replicated. Each rank updates its own parameters with AdamW. Global norm
  clipping counts replicas and tied parameters once.

One default process group represents **one model**, not multiple data-parallel
replicas. Whole attention heads are distributed as evenly as possible across
ranks; the head count must be at least the TP world size. For the 355M model
on three ranks, the 16 heads split 6/5/5, and the 4096 MLP features split
1366/1365/1365. This preserves the model but introduces workload imbalance.
All ranks must execute
the same forward/backward calls. Rank 0 reads and broadcasts training/validation
batches; all ranks evaluate the same HellaSwag examples. Sampling broadcasts
rank 0's chosen tokens, and only rank 0 prints them.

Initialization uses the same seed on all ranks and initializes one full CPU
weight tensor at a time before slicing it, avoiding identical repeated shards.
This prioritizes simplicity over startup memory efficiency. There is no dropout
in the original model, so no separate dropout RNG streams are required.

## Running

On the two machines, from identical copies of this branch:

```bash
# Node 0 (192.168.2.1)
bash launch_distributed.sh 0
# Node 1
bash launch_distributed.sh 1
```

The launcher uses the local `gpt/bin/python` when available. Override `PYTHON`,
`MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NPROC_PER_NODE`, or NCCL environment
settings as needed. Its default network interface is `enp2s0`; NCCL selects
the available transport. The launcher never kills processes occupying a port.

Training defaults to FP32 (appropriate for the GTX 1080 Ti), microbatch 1,
sequence length 1024, and global batch 2048 tokens. This means **two accumulation
steps regardless of TP world size**. TP reductions remain necessary on every
microstep. `--dtype bfloat16` is optional on GPUs with native BF16 support.

For a short training-only run, on both nodes:

```bash
bash launch_distributed.sh 0 --skip-eval --max-steps 3 --log-dir log/tp-smoke
# Use node rank 1 on the second machine.
```

On a single machine with two GPUs:

```bash
gpt/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 train_gpt2.py
```

The dataset is required on rank 0 at `edu_fineweb10B` (or `--data-root`). HellaSwag
evaluation uses the existing loader on each rank and requires the same examples
in the same order. `--skip-eval` skips validation, HellaSwag, and text generation.
Model dimensions, batch sizes, sequence length, and output directory can be
overridden; see `train_gpt2.py --help`.

## Model API and checkpoints

`GPT(config, tp_rank, tp_world_size)` requires an initialized matching process
group when world size exceeds one. `GPT(config)` also works without distributed
initialization. Importing `train_gpt2` does not start training.

`model(tokens, targets)` returns **local vocabulary logits** and a replicated
mean loss. `model(tokens)` returns gathered full logits for evaluation/generation.
Pass `gather_output=True` explicitly if full logits are needed with targets.

All ranks call `full_state_dict(model)`; rank 0 receives a conventional CPU state
dict, including all shards and conventional full `[Q, K, V]` ordering. Training
saves it after the final update and every 5000 steps. Checkpoints store config as
a plain dictionary and can be loaded with `weights_only=True`:

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model = GPT(GPTConfig(**checkpoint["config"]), tp_rank, tp_world_size)
load_full_state_dict(model, checkpoint["model"])
```

Every rank needs the full checkpoint when using this simple loader. Checkpoints
are portable across compatible TP sizes, including a single GPU. They contain
model weights, config, and step, not optimizer/RNG/dataloader state; they are not
exact training-resume snapshots. `GPT.from_pretrained` also accepts TP rank and
world size, using the existing optional Hugging Face loader before partitioning.
Old checkpoints saved from only rank 0 of the broken TP implementation cannot
recover missing shards.

## Validation

```bash
gpt/bin/python -m unittest -v test_tensor_parallel
```

The tests use an independent dense `nn.Linear`/`nn.Embedding` reference, two and
three CPU/Gloo processes, divisible and uneven vocabularies, uneven attention
heads and MLP partitions, nonzero biases,
gradient accumulation, clipping, multiple AdamW steps, checkpoint round trips,
extreme-logit cross-entropy, and collective counts. They run the real training,
validation, evaluation, and sampling paths with small local fixtures. A separate
single-GPU correctness check runs when CUDA is available. Multi-GPU NCCL and
cross-machine throughput still require testing on the target machines.
