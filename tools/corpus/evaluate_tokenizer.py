#!/usr/bin/env python3
"""Evaluate a saved Tata tokenizer on fixed Indonesian edge-case probes.

The probes are evaluation-only and are never added to tokenizer training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

try:
    from tools.corpus.train_tokenizer import CharacterBPE
except ModuleNotFoundError:
    from train_tokenizer import CharacterBPE


DEFAULT_PROBES = [
    "membaca, membacakan, dibacakan, pembacaan",
    "berlari, berlarian, permainan, mempermainkan",
    "kata-kata dan rumah-rumah",
    "Rp10.000,50; Rp125.000; diskon 12,5%",
    "Tanggal 28/11/2026 pukul 20.30 WIB",
    "Tata AI: apa kabar? Tolong jelaskan dengan sederhana!",
    "AI generatif offline-first tanpa API",
    "Kediri, Jawa Timur — Indonesia.",
    "Saya tidak tahu; mari kita periksa sumbernya.",
    "Emoji uji: 😀",
]


def evaluate_text(tokenizer: CharacterBPE, text: str, probe_id: str = "probe-1") -> dict:
    """Return aggregate metrics only; never copy probe content into reports."""
    tokens = tokenizer.encode(text)
    unknown_count = sum(token == "<unk>" for token in tokens)
    return {
        "probe_id": probe_id,
        "characters": len(text),
        "tokens": len(tokens),
        "unknown_tokens": unknown_count,
        "unknown_rate": unknown_count / len(text) if text else 0.0,
        "tokens_per_character": len(tokens) / len(text) if text else 0.0,
        "round_trip_exact": tokenizer.decode(tokens) == text,
    }


def build_report(tokenizer_path: Path, custom_texts: list[str] | None = None) -> dict:
    tokenizer = CharacterBPE.load(tokenizer_path)
    probes = custom_texts if custom_texts else DEFAULT_PROBES
    if not probes or any(not isinstance(text, str) or not text for text in probes):
        raise ValueError("Setiap teks uji harus berupa string yang tidak kosong")
    results = [
        evaluate_text(tokenizer, text, probe_id=f"probe-{index + 1:03d}")
        for index, text in enumerate(probes)
    ]
    total_chars = sum(item["characters"] for item in results)
    total_tokens = sum(item["tokens"] for item in results)
    total_unknown = sum(item["unknown_tokens"] for item in results)
    return {
        "format": "tata-tokenizer-evaluation",
        "format_version": 1,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "tokenizer_vocab_size": len(tokenizer.vocab),
        "probe_count": len(results),
        "summary": {
            "characters": total_chars,
            "tokens": total_tokens,
            "unknown_tokens": total_unknown,
            "unknown_rate": total_unknown / total_chars if total_chars else 0.0,
            "tokens_per_character": total_tokens / total_chars if total_chars else 0.0,
            "exact_round_trip_count": sum(item["round_trip_exact"] for item in results),
        },
        "probes": results,
        "limitations": [
            "Probe tetap ini bukan pengganti evaluasi pada korpus besar yang berizin.",
            "Rasio token rendah tidak membuktikan kualitas bahasa atau kemampuan model.",
            "Teks probe tidak digunakan untuk melatih tokenizer atau model.",
        ],
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--text", action="append", default=[], help="teks uji tambahan; dapat diulang")
    parser.add_argument("--output", type=Path, help="opsional: simpan laporan JSON")
    args = parser.parse_args()
    try:
        report = build_report(args.tokenizer, args.text or None)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            atomic_write_json(args.output, report)
        print(rendered)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
