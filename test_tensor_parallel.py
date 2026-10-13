"""Run with: gpt/bin/python -m unittest -v test_tensor_parallel

CPU/Gloo multiprocess tests need no dataset, downloads, or extra test libraries.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from train_gpt2 import GPT, GPTConfig, Block
from tensor_parallel import (
    TPLinear, TPLinearReduce, VocabParallelEmbedding, clip_grad_norm_tp,
    full_state_dict, load_full_state_dict, shard_tensor,
    vocab_parallel_cross_entropy,
)


class DenseGPT(GPT):
    """Plain nn.Linear/nn.Embedding model, ordinary autograd and cross-entropy."""

    def __init__(self, config):
        super().__init__(config)
        for module in list(self.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, (TPLinear, TPLinearReduce)):
                    replacement = nn.Linear(child.full_shape[1], child.full_shape[0],
                                            bias=child.bias is not None)
                elif isinstance(child, VocabParallelEmbedding):
                    replacement = nn.Embedding(config.vocab_size, config.n_embd)
                else:
                    continue
                replacement.load_state_dict(child.state_dict())
                setattr(module, name, replacement)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, tokens, targets=None):
        x = self.transformer.wte(tokens) + self.transformer.wpe(
            torch.arange(tokens.size(1), device=tokens.device))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.flatten(0, 1), targets.flatten())
        return logits, loss


def compare_parameters(parallel, dense, gradients=False):
    reference = dict(dense.named_parameters())
    for name, parameter in parallel.named_parameters():
        expected = reference[name].grad if gradients else reference[name].detach()
        actual = parameter.grad if gradients else parameter.detach()
        if name == "transformer.wte.weight":
            padding = parameter.size(0) * parallel.tp_world_size - expected.size(0)
            expected = F.pad(expected, (0, 0, 0, padding))
        expected = shard_tensor(expected, getattr(parameter, "tp_dim", None),
                                parallel.tp_rank, parallel.tp_world_size,
                                getattr(parameter, "tp_stride", 1))
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-7,
                                   msg=lambda message: f"{name}: {message}")


def distributed_worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}",
                            rank=rank, world_size=world_size)
    try:
        for vocab_size in (36, 37):
            config = GPTConfig(block_size=8, vocab_size=vocab_size, n_layer=2,
                               n_head=6, n_embd=24)
            torch.manual_seed(17)
            base = GPT(config)
            torch.manual_seed(17)
            parallel = GPT(config, rank, world_size)
            if vocab_size == 36:
                # Initialization itself must yield the same full model at all TP sizes.
                compare_parameters(parallel, base)
            dense = DenseGPT(config).double()
            dense.load_state_dict(base.state_dict())
            parallel = parallel.double()
            # Nonzero biases catch bias being incorrectly summed world_size times.
            with torch.no_grad():
                for name, parameter in dense.named_parameters():
                    if name.endswith("bias"):
                        parameter.uniform_(-0.1, 0.1)
            load_full_state_dict(parallel, dense.state_dict())
            assert parallel.transformer.wte.weight is parallel.lm_head.weight
            opt_tp = torch.optim.AdamW(parallel.parameters(), lr=0.002, eps=1e-6)
            opt_dense = torch.optim.AdamW(dense.parameters(), lr=0.002, eps=1e-6)
            for step in range(3):
                opt_tp.zero_grad()
                opt_dense.zero_grad()
                for microstep in range(2):
                    tokens = torch.arange(16).reshape(2, 8).add(step * 5 + microstep) % vocab_size
                    targets = (tokens + 9) % vocab_size
                    # Training must not all-gather any activations/logits.
                    with patch.object(dist, "all_gather", side_effect=AssertionError("Training all-gather")):
                        _, loss_tp = parallel(tokens, targets)
                        logits_dense, loss_dense = dense(tokens, targets)
                        torch.testing.assert_close(loss_tp, loss_dense, atol=1e-10, rtol=1e-9)
                        (loss_tp / 2).backward()
                    (loss_dense / 2).backward()
                    with torch.no_grad():
                        logits_tp, _ = parallel(tokens)
                        torch.testing.assert_close(logits_tp, logits_dense, atol=1e-10, rtol=1e-9)
                compare_parameters(parallel, dense, gradients=True)
                norm_tp = clip_grad_norm_tp(parallel, 0.1)
                norm_dense = torch.nn.utils.clip_grad_norm_(dense.parameters(), 0.1)
                torch.testing.assert_close(norm_tp, norm_dense, atol=1e-10, rtol=1e-9)
                compare_parameters(parallel, dense, gradients=True)
                opt_tp.step()
                opt_dense.step()
                compare_parameters(parallel, dense)
            state = full_state_dict(parallel)
            if rank == 0:
                for name, expected in dense.state_dict().items():
                    torch.testing.assert_close(state[name], expected, atol=2e-9, rtol=2e-7)
                checkpoint_path = rendezvous + ".pt"
                torch.save(state, checkpoint_path)
            dist.barrier()
            restored = GPT(config, rank, world_size).double()
            load_full_state_dict(restored, torch.load(rendezvous + ".pt", weights_only=True))
            compare_parameters(restored, dense)
            # A gathered-logits loss is also differentiable (evaluation/fine-tuning).
            restored.zero_grad()
            dense.zero_grad()
            logits, _ = restored(tokens)
            F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
            dense(tokens, targets)[1].backward()
            compare_parameters(restored, dense, gradients=True)

        # Exactly two reductions forward and two backward per transformer block.
        block = Block(config, rank, world_size).double()
        block.apply(parallel._init_weights)
        x = torch.randn(2, 4, config.n_embd, dtype=torch.float64, requires_grad=True)
        with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduction:
            y = block(x)
            assert reduction.call_count == 2
            y.sum().backward()
            assert reduction.call_count == 4

        # Extreme logits exercise stable distributed log-sum-exp and its gradient.
        scores = torch.linspace(-1000, 1000, 36, dtype=torch.float64).repeat(4, 1)
        targets = torch.tensor([0, 12, 24, 35])
        dense_scores = scores.clone().requires_grad_()
        local_scores = scores.chunk(world_size, -1)[rank].clone().requires_grad_()
        loss = vocab_parallel_cross_entropy(local_scores, targets, rank, world_size).mean()
        expected_loss = F.cross_entropy(dense_scores, targets)
        torch.testing.assert_close(loss, expected_loss)
        loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(local_scores.grad, dense_scores.grad.chunk(world_size, -1)[rank])
    finally:
        dist.destroy_process_group()


class TensorParallelTests(unittest.TestCase):
    def test_distributed_equivalence(self):
        for world_size in (2, 3):
            with self.subTest(world_size=world_size), tempfile.TemporaryDirectory() as directory:
                mp.spawn(distributed_worker, args=(world_size, os.path.join(directory, "rendezvous")),
                         nprocs=world_size, join=True)

    def test_training_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / "train_000.npy", np.arange(256) % 37)
            result = subprocess.run([
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                "train_gpt2.py", "--device", "cpu", "--skip-eval", "--max-steps", "2",
                "--batch-size", "1", "--total-batch-size", "16", "--sequence-length", "8",
                "--n-layer", "2", "--n-head", "6", "--n-embd", "24", "--vocab-size", "37",
                "--data-root", directory, "--log-dir", directory,
            ], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("gradient accumulation steps: 2", result.stdout)
            self.assertEqual(len((Path(directory) / "log.txt").read_text().splitlines()), 2)
            checkpoint = torch.load(Path(directory) / "model_00001.pt", weights_only=True)
            dense = DenseGPT(GPTConfig(**checkpoint["config"]))
            dense.load_state_dict(checkpoint["model"])
            self.assertEqual(checkpoint["step"], 1)

    def test_evaluation_and_generation_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            for split in ("train", "val"):
                np.save(Path(directory) / f"{split}_000.npy", np.arange(2048) % 60)
            result = subprocess.run([
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                "test_tensor_parallel.py", "--evaluation-worker", "--device", "cpu", "--max-steps", "1",
                "--total-batch-size", "64", "--sequence-length", "32", "--n-layer", "1",
                "--n-head", "6", "--n-embd", "24", "--vocab-size", "64",
                "--data-root", directory, "--log-dir", directory,
            ], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("validation loss:", result.stdout)
            self.assertIn("HellaSwag accuracy:", result.stdout)
            self.assertEqual(result.stdout.count("rank 0 sample"), 4)
            self.assertNotIn("rank 1 sample", result.stdout)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_single_gpu(self):
        torch.manual_seed(42)
        config = GPTConfig(block_size=8, vocab_size=37, n_layer=2, n_head=6, n_embd=24)
        parallel = GPT(config).cuda().double()
        dense = DenseGPT(config).cuda().double()
        dense.load_state_dict(parallel.state_dict())
        tokens = torch.arange(8, device="cuda").unsqueeze(0)
        _, loss = parallel(tokens, tokens)
        _, expected = dense(tokens, tokens)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        expected.backward()
        compare_parameters(parallel, dense, gradients=True)


if __name__ == "__main__":
    if "--evaluation-worker" in sys.argv:
        # Exercise the actual training/eval/generation loop using local fixtures.
        # Only the external dataset and tokenizer are substituted, not model code.
        sys.argv.remove("--evaluation-worker")
        tokenizer = types.SimpleNamespace(
            n_vocab=60, encode=lambda text: [1, 2, 3, 4],
            decode=lambda tokens: " ".join(str(token) for token in tokens))
        sys.modules["tiktoken"] = types.SimpleNamespace(get_encoding=lambda name: tokenizer)
        tokens = torch.arange(32).reshape(4, 8)
        mask = torch.ones_like(tokens)
        mask[:, :4] = 0
        sys.modules["hellaswag"] = types.SimpleNamespace(
            iterate_examples=lambda split: iter([0, 1]),
            render_example=lambda example: (None, tokens + example, mask, example))
        from train_gpt2 import main
        main()
    else:
        unittest.main()
