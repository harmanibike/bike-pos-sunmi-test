#!/usr/bin/env python3
"""Train the small Tata decoder-only Transformer locally from random weights.

Requires PyTorch 2.x. No network access, API, pretrained weights, or test-set loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

from transformer_config import TransformerConfig
from training_data import load_reviewed_sequences


def build_model_class(torch):
    nn = torch.nn
    functional = torch.nn.functional

    class TransformerBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.norm_attention = nn.LayerNorm(config.hidden_size)
            self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.use_bias)
            self.attention_output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
            self.norm_mlp = nn.LayerNorm(config.hidden_size)
            self.mlp_up = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
            self.mlp_down = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)
            self.heads = config.num_heads
            self.head_dim = config.head_dim

        def forward(self, value):
            batch, length, width = value.shape
            normalized = self.norm_attention(value)
            qkv = self.qkv(normalized).view(batch, length, 3, self.heads, self.head_dim)
            query, key, content = qkv.unbind(dim=2)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            content = content.transpose(1, 2)
            attended = functional.scaled_dot_product_attention(
                query, key, content, is_causal=True
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
            value = value + self.attention_output(attended)
            hidden = self.norm_mlp(value)
            hidden = functional.gelu(self.mlp_up(hidden))
            return value + self.mlp_down(hidden)

    class TataTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
            self.blocks = nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.num_layers)]
            )
            self.final_norm = nn.LayerNorm(config.hidden_size)
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
            if config.tie_embeddings:
                self.lm_head.weight = self.token_embedding.weight
            self.context_length = config.context_length
            self.apply(self._initialize)

        @staticmethod
        def _initialize(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, input_ids):
            _, length = input_ids.shape
            if length > self.context_length:
                raise ValueError("Panjang input melebihi context_length")
            positions = torch.arange(length, device=input_ids.device)
            hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
            for block in self.blocks:
                hidden = block(hidden)
            return self.lm_head(self.final_norm(hidden))

    return TataTransformer


def load_resume_checkpoint(torch, source: Path, device):
    """Load only safe tensor/primitive checkpoint data; never unpickle arbitrary objects."""
    try:
        checkpoint = torch.load(
            source, map_location=device, weights_only=True
        )
    except TypeError as error:
        raise RuntimeError(
            "Versi PyTorch tidak mendukung pemuatan checkpoint aman (weights_only=True)"
        ) from error
    except Exception as error:
        raise ValueError("Checkpoint tidak dapat dibaca dengan mode aman") from error

    if not isinstance(checkpoint, dict):
        raise ValueError("Format checkpoint tidak valid: harus berupa dictionary")
    if checkpoint.get("format") != "tata-transformer-checkpoint":
        raise ValueError("Format checkpoint bukan milik Tata Transformer")
    if checkpoint.get("format_version") != 1:
        raise ValueError("Versi format checkpoint tidak didukung")
    if checkpoint.get("checkpoint_role") != "latest_resume":
        raise ValueError("Checkpoint bukan checkpoint untuk melanjutkan training")
    required = {
        "step", "config", "model", "optimizer", "train_meta", "validation_meta"
    }
    if not required.issubset(checkpoint):
        raise ValueError("Checkpoint tidak lengkap: ada bagian wajib yang hilang")
    if not isinstance(checkpoint["model"], dict) or not isinstance(checkpoint["optimizer"], dict):
        raise ValueError("State model atau optimizer pada checkpoint tidak valid")
    return checkpoint


def atomic_save(torch, payload, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=destination.name + ".",
            suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def batch_from_sequences(torch, sequences, batch_size, context_length, device, rng):
    selected = [rng.choice(sequences) for _ in range(batch_size)]
    inputs, targets = [], []
    for sequence in selected:
        start = rng.randrange(0, len(sequence) - context_length)
        window = sequence[start:start + context_length + 1]
        inputs.append(window[:-1])
        targets.append(window[1:])
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def select_validation_windows(sequences, context_length, max_batches=32):
    if context_length < 1 or max_batches < 1:
        raise ValueError("context_length dan max_batches harus positif")
    windows = []
    for sequence in sequences:
        for start in range(0, len(sequence) - context_length, context_length):
            window = sequence[start:start + context_length + 1]
            if len(window) == context_length + 1:
                windows.append(window)
    if len(windows) <= max_batches:
        return windows
    if max_batches == 1:
        return [windows[0]]
    indexes = [
        round(index * (len(windows) - 1) / (max_batches - 1))
        for index in range(max_batches)
    ]
    return [windows[index] for index in indexes]


def evaluate(torch, model, sequences, config, device, max_batches=32):
    windows = select_validation_windows(
        sequences, config.context_length, max_batches
    )
    if not windows:
        raise ValueError("Validation tidak memiliki sequence cukup panjang untuk context_length")
    model.eval()
    losses = []
    with torch.no_grad():
        for window in windows:
            x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([window[1:]], dtype=torch.long, device=device)
            logits = model(x)
            losses.append(torch.nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size), y.reshape(-1)
            ).item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    if not math.isfinite(mean_loss):
        raise ValueError("Validation loss tidak finite; training dihentikan")
    return mean_loss


def best_checkpoint_path(output: Path) -> Path:
    return output.with_name(output.stem + ".best" + output.suffix)


def clone_state_to_cpu(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def early_stop_reached(patience: int, evaluations_without_improvement: int) -> bool:
    return patience > 0 and evaluations_without_improvement >= patience


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-register", type=Path, required=True,
                        help="CSV register sumber dengan izin training yang disetujui")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--context-length", type=int, default=256,
        help="panjang konteks eksperimen; boleh lebih kecil untuk korpus pilot",
    )
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="jumlah evaluasi tanpa perbaikan sebelum berhenti; 0 menonaktifkan")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_interval, args.checkpoint_interval) <= 0:
        parser.error("steps, batch-size, eval-interval, dan checkpoint-interval harus positif")
    if args.learning_rate <= 0:
        parser.error("learning-rate harus positif")
    if args.early_stopping_patience < 0:
        parser.error("early-stopping-patience tidak boleh negatif")
    if args.context_length < 2:
        parser.error("context-length minimal 2 token")

    try:
        import torch
    except ImportError:
        parser.error("PyTorch 2.x belum terpasang. Pasang sesuai perangkat pelatihan lokal.")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA diminta tetapi tidak tersedia")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    train_sequences, train_meta = load_reviewed_sequences(
        args.corpus, args.tokenizer, "train", args.source_register
    )
    validation_sequences, validation_meta = load_reviewed_sequences(
        args.corpus, args.tokenizer, "validation", args.source_register
    )
    config = TransformerConfig(
        vocab_size=train_meta["vocab_size"],
        context_length=args.context_length,
    )
    config.validate()
    if config.vocab_size <= 0:
        parser.error("Konfigurasi vocabulary tidak valid")
    if validation_meta["vocab_size"] != train_meta["vocab_size"]:
        parser.error("Vocabulary tokenizer train dan validation tidak konsisten")
    minimum = config.context_length + 1
    train_sequences = [sequence for sequence in train_sequences if len(sequence) >= minimum]
    validation_sequences = [sequence for sequence in validation_sequences if len(sequence) >= minimum]
    if not train_sequences or not validation_sequences:
        parser.error(
            f"Data train/validation belum cukup panjang. Minimal {minimum} token per contoh; "
            "pastikan kedua split berisi data yang sudah lulus tinjauan."
        )

    random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_class = build_model_class(torch)
    model = model_class(config).to(device)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = config.parameter_count()
    if actual_parameters != expected_parameters:
        parser.error(
            f"Hitungan parameter implementasi ({actual_parameters}) tidak cocok "
            f"dengan estimator ({expected_parameters})"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_step = 0
    best_validation = math.inf
    best_model_state = None
    best_step = None
    evaluations_without_improvement = 0
    if args.resume:
        if not args.output.exists():
            parser.error("--resume dipakai tetapi checkpoint tidak ditemukan")
        checkpoint = load_resume_checkpoint(torch, args.output, device)
        if checkpoint.get("config") != config.report():
            parser.error("Konfigurasi checkpoint berbeda")
        if checkpoint.get("train_meta", {}).get("corpus_sha256") != train_meta["corpus_sha256"]:
            parser.error("Hash korpus train berubah; checkpoint tidak dapat dilanjutkan")
        if checkpoint.get("train_meta", {}).get("tokenizer_sha256") != train_meta["tokenizer_sha256"]:
            parser.error("Tokenizer berubah; checkpoint tidak dapat dilanjutkan")
        if checkpoint.get("train_meta", {}).get("source_register_sha256") != train_meta["source_register_sha256"]:
            parser.error("Register izin sumber berubah; checkpoint tidak dapat dilanjutkan")
        if checkpoint.get("validation_meta", {}).get("corpus_sha256") != validation_meta["corpus_sha256"]:
            parser.error("Hash korpus validation berubah; checkpoint tidak dapat dilanjutkan")
        if checkpoint.get("validation_meta", {}).get("source_register_sha256") != validation_meta["source_register_sha256"]:
            parser.error("Register izin sumber validation berubah; checkpoint tidak dapat dilanjutkan")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_validation = float(checkpoint.get("best_validation_loss", math.inf))
        best_model_state = checkpoint.get("best_model")
        best_step = checkpoint.get("best_step")
        evaluations_without_improvement = int(checkpoint.get("evaluations_without_improvement", 0))
        if best_model_state is None:
            best_model_state = clone_state_to_cpu(model.state_dict())
            best_step = start_step
            best_validation = math.inf
            evaluations_without_improvement = 0
        if "python_rng_state" in checkpoint:
            rng.setstate(checkpoint["python_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    model.train()
    train_loss_sum = 0.0
    train_loss_count = 0
    stopped_early = False
    for step in range(start_step + 1, start_step + args.steps + 1):
        inputs, targets = batch_from_sequences(
            torch, train_sequences, args.batch_size, config.context_length, device, rng
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), targets.reshape(-1)
        )
        if not math.isfinite(float(loss.item())):
            raise RuntimeError(f"Training loss tidak finite pada step {step}; training dihentikan")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(gradient_norm)):
            raise RuntimeError(f"Gradient norm tidak finite pada step {step}; training dihentikan")
        optimizer.step()
        train_loss_sum += float(loss.item())
        train_loss_count += 1

        should_evaluate = step % args.eval_interval == 0 or step == start_step + args.steps
        stop_early = False
        if should_evaluate:
            validation_loss = evaluate(
                torch, model, validation_sequences, config, device
            )
            improved = validation_loss < best_validation
            if improved:
                best_validation = validation_loss
                best_step = step
                best_model_state = clone_state_to_cpu(model.state_dict())
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            mean_train_loss = train_loss_sum / max(train_loss_count, 1)
            generalization_gap = validation_loss - mean_train_loss
            train_loss_sum = 0.0
            train_loss_count = 0
            print(json.dumps({
                "step": step,
                "train_loss_mean_since_last_eval": round(mean_train_loss, 6),
                "validation_loss": round(validation_loss, 6),
                "validation_minus_train_loss": round(generalization_gap, 6),
                "best_validation_loss": round(best_validation, 6),
                "best_step": best_step,
                "evaluations_without_improvement": evaluations_without_improvement,
                "perplexity": round(math.exp(min(validation_loss, 20)), 4),
                "device": device,
            }, ensure_ascii=False), flush=True)
            stop_early = early_stop_reached(
                args.early_stopping_patience, evaluations_without_improvement
            )
            if improved:
                best_payload = {
                    "format": "tata-transformer-checkpoint",
                    "format_version": 1,
                    "checkpoint_role": "best_validation",
                    "step": step,
                    "seed": args.seed,
                    "config": config.report(),
                    "model": best_model_state,
                    "optimizer": optimizer.state_dict(),
                    "train_meta": train_meta,
                    "validation_meta": validation_meta,
                    "best_validation_loss": best_validation,
                    "best_step": best_step,
                    "training": {"learning_rate": args.learning_rate,
                                 "batch_size": args.batch_size,
                                 "device": device,
                                 "initialization": "random",
                                 "pretrained_weights": False,
                                 "early_stopping_patience": args.early_stopping_patience},
                }
                atomic_save(torch, best_payload, best_checkpoint_path(args.output))

        if (step % args.checkpoint_interval == 0
                or step == start_step + args.steps or stop_early):
            payload = {
                "format": "tata-transformer-checkpoint",
                "format_version": 1,
                "checkpoint_role": "latest_resume",
                "step": step,
                "seed": args.seed,
                "config": config.report(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "python_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                "train_meta": train_meta,
                "validation_meta": validation_meta,
                "best_validation_loss": best_validation,
                "best_step": best_step,
                "best_model": best_model_state,
                "evaluations_without_improvement": evaluations_without_improvement,
                "training": {
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "device": device,
                    "initialization": "random",
                    "pretrained_weights": False,
                    "early_stopping_patience": args.early_stopping_patience,
                },
            }
            atomic_save(torch, payload, args.output)

        if stop_early:
            stopped_early = True
            print(json.dumps({"status": "early_stopping", "step": step,
                              "best_step": best_step,
                              "best_validation_loss": best_validation}, ensure_ascii=False), flush=True)
            break

    print(json.dumps({
        "status": "early_stopped" if stopped_early else "training_finished",
        "steps_completed": step,
        "checkpoint": str(args.output),
        "best_checkpoint": str(best_checkpoint_path(args.output)),
        "best_step": best_step,
        "best_validation_loss": best_validation,
        "parameters": actual_parameters,
        "train_records": train_meta["records"],
        "validation_records": validation_meta["records"],
        "test_split_loaded": False,
        "note": "Hasil wajib dievaluasi; checkpoint belum berarti model siap dipakai.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
