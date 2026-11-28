#!/usr/bin/env python3
"""Export a reviewed Tata best-validation checkpoint to an experimental ONNX artifact.

This does not train a model and does not modify the Android app or existing GGUF path.
Requires PyTorch 2.x and ONNX. No network/API access is used by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))

from transformer_config import TransformerConfig
from train_transformer import build_model_class
from training_data import tokenizer_vocabulary
from tools.corpus.train_tokenizer import CharacterBPE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_best_checkpoint(torch, path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("PyTorch wajib mendukung weights_only=True") from error
    except Exception as error:
        raise ValueError("Checkpoint gagal dibaca secara aman") from error
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint harus berupa dictionary")
    if payload.get("format") != "tata-transformer-checkpoint" or payload.get("format_version") != 1:
        raise ValueError("Format checkpoint Tata tidak dikenal")
    if payload.get("checkpoint_role") != "best_validation":
        raise ValueError("Ekspor hanya menerima checkpoint best_validation")
    if not isinstance(payload.get("model"), dict):
        raise ValueError("State model tidak tersedia")
    if not isinstance(payload.get("config"), dict) or not isinstance(payload["config"].get("config"), dict):
        raise ValueError("Konfigurasi model tidak tersedia")
    if not isinstance(payload.get("train_meta"), dict) or not isinstance(payload.get("validation_meta"), dict):
        raise ValueError("Metadata train/validation tidak tersedia")
    training = payload.get("training")
    if not isinstance(training, dict) or training.get("initialization") != "random" or training.get("pretrained_weights") is not False:
        raise ValueError("Asal bobot tidak dapat diverifikasi sebagai random initialization")
    return payload


def build_export_manifest(checkpoint: dict, checkpoint_path: Path, tokenizer_path: Path,
                          tokenizer: CharacterBPE, vocab: list[str], onnx_path: Path) -> dict:
    config_report = checkpoint["config"]
    return {
        "format": "tata-onnx-model-package",
        "format_version": 1,
        "status": "experimental_not_android_validated",
        "model": {
            "file": onnx_path.name,
            "sha256": sha256_file(onnx_path),
            "input": {"name": "input_ids", "dtype": "int64", "shape": ["batch", "sequence"]},
            "output": {"name": "logits", "dtype": "float32", "shape": ["batch", "sequence", len(vocab)]},
            "opset": 17,
        },
        "architecture": config_report,
        "tokenizer": {
            "format": "tata-character-bpe",
            "format_version": 1,
            "file": tokenizer_path.name,
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": len(vocab),
            "id_to_token": vocab,
            "special_tokens": tokenizer.to_dict()["special_tokens"],
            "tokenization_note": "CharacterBPE encode applies merges in saved order; Kotlin JSON loader and synthetic cross-language parity fixture tests exist.",
        },
        "provenance": {
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_role": checkpoint.get("checkpoint_role"),
            "step": checkpoint.get("step"),
            "seed": checkpoint.get("seed"),
            "train_corpus_sha256": checkpoint["train_meta"].get("corpus_sha256"),
            "validation_corpus_sha256": checkpoint["validation_meta"].get("corpus_sha256"),
            "source_register_sha256": checkpoint["train_meta"].get("source_register_sha256"),
            "initialization": "random",
            "pretrained_weights": False,
        },
        "limitations": [
            "Experimental export; not yet validated against ONNX Runtime Android.",
            "Tokenizer Kotlin parity is covered by a synthetic shared fixture; broader corpus parity and Android runtime validation remain open.",
            "No claim of production training quality or Android device performance.",
        ],
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def export(checkpoint_path: Path, tokenizer_path: Path, output_path: Path) -> dict:
    try:
        import torch
        import onnx
    except ImportError as error:
        raise RuntimeError("Instal PyTorch 2.x dan paket onnx untuk ekspor eksperimental") from error

    checkpoint = load_best_checkpoint(torch, checkpoint_path)
    config_values = checkpoint["config"]["config"]
    config = TransformerConfig(**config_values)
    config.validate()
    tokenizer = CharacterBPE.load(tokenizer_path)
    vocab, token_to_id = tokenizer_vocabulary(tokenizer)
    if len(vocab) != config.vocab_size or len(token_to_id) != config.vocab_size:
        raise ValueError("Vocabulary tokenizer tidak sama dengan vocab_size checkpoint")
    expected_tokenizer_hash = checkpoint["train_meta"].get("tokenizer_sha256")
    actual_tokenizer_hash = sha256_file(tokenizer_path)
    if not expected_tokenizer_hash or expected_tokenizer_hash != actual_tokenizer_hash:
        raise ValueError("Hash tokenizer tidak cocok dengan metadata checkpoint training")
    train_register_hash = checkpoint["train_meta"].get("source_register_sha256")
    validation_register_hash = checkpoint["validation_meta"].get("source_register_sha256")
    if not train_register_hash or train_register_hash != validation_register_hash:
        raise ValueError("Register sumber train dan validation tidak cocok")
    if checkpoint["config"].get("derived", {}).get("total_parameters") != config.parameter_count():
        raise ValueError("Parameter count konfigurasi checkpoint tidak konsisten")

    model_class = build_model_class(torch)
    model = model_class(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    sample = torch.ones((1, min(4, config.context_length)), dtype=torch.long)
    try:
        torch.onnx.export(
            model, sample, str(temporary),
            input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "logits": {0: "batch", 1: "sequence"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest = build_export_manifest(
        checkpoint, checkpoint_path, tokenizer_path, tokenizer, vocab, output_path
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    atomic_json(manifest_path, manifest)
    return {"model": str(output_path), "manifest": str(manifest_path),
            "sha256": manifest["model"]["sha256"], "vocab_size": len(vocab)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="checkpoint .best.pt dengan role best_validation")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="tujuan file ONNX")
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.checkpoint, args.tokenizer, args.output),
                         ensure_ascii=False, indent=2))
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
