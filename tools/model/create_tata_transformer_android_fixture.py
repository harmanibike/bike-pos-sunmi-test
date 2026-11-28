#!/usr/bin/env python3
"""Create a tiny random-initialized Tata Transformer ONNX model for Android CI.

This is a software compatibility fixture only. It is not trained, contains no
user data, and must never be represented as a production language model.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from tools.corpus.train_tokenizer import CharacterBPE
from tools.model.export_transformer_onnx import export
from tools.model.train_transformer import build_model_class
from tools.model.transformer_config import TransformerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--parity-output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.parity_output.parent.mkdir(parents=True, exist_ok=True)

    config = TransformerConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        context_length=8,
        tie_embeddings=True,
    )
    config.validate()

    with tempfile.TemporaryDirectory(prefix="tata-onnx-ci-") as directory:
        work = Path(directory)
        tokenizer_path = work / "tokenizer.json"
        checkpoint_path = work / "best.pt"
        CharacterBPE(["a", "b", "c", "d"], []).save(tokenizer_path)

        torch.manual_seed(20261128)
        model = build_model_class(torch)(config)
        tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        torch.save({
            "format": "tata-transformer-checkpoint",
            "format_version": 1,
            "checkpoint_role": "best_validation",
            "step": 0,
            "seed": 20261128,
            "config": config.report(),
            "model": model.state_dict(),
            "train_meta": {
                "corpus_sha256": "synthetic-ci-only",
                "tokenizer_sha256": tokenizer_hash,
                "source_register_sha256": "synthetic-ci-register",
            },
            "validation_meta": {
                "corpus_sha256": "synthetic-ci-only",
                "source_register_sha256": "synthetic-ci-register",
            },
            "training": {
                "initialization": "random",
                "pretrained_weights": False,
            },
        }, checkpoint_path)

        result = export(checkpoint_path, tokenizer_path, args.output)
        model.eval()
        parity = {}
        with torch.no_grad():
            samples = {
                "1x4": [[4, 5, 6, 7]],
                "2x5": [[4, 5, 6, 7, 4], [5, 6, 7, 4, 5]],
            }
            for name, values in samples.items():
                parity[name] = {
                    "input_ids": values,
                    "logits": model(torch.tensor(values, dtype=torch.long)).tolist(),
                }
        import json
        args.parity_output.write_text(
            json.dumps(parity, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        print(f"Generated synthetic Tata Transformer ONNX fixture: {result['model']}")
        print(f"Generated PyTorch parity reference: {args.parity_output}")
        print(f"SHA-256: {result['sha256']}")


if __name__ == "__main__":
    main()
