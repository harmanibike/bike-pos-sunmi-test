#!/usr/bin/env python3
"""Eksperimen byte-independent character BPE sederhana untuk korpus Tata AI.

Ini tokenizer riset awal, bukan tokenizer produksi. Tidak memakai layanan/API.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from tools.corpus.validate_corpus import validate_file
    from tools.corpus.validate_source_register import validate_register
except ModuleNotFoundError:
    from validate_corpus import validate_file
    from validate_source_register import validate_register

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
ALLOWED_LICENSES = {
    "public_domain", "cc0", "cc_by", "cc_by_sa",
    "owned_or_explicitly_permitted", "explicit_permission",
}


class CharacterBPE:
    def __init__(self, alphabet: list[str], merges: list[tuple[str, str]]):
        self.alphabet = list(alphabet)
        self.merges = list(merges)
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = set(SPECIAL_TOKENS) | set(self.alphabet)
        for left, right in self.merges:
            self.vocab.add(left + right)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "CharacterBPE":
        corpus = [list(text) for text in texts if text]
        if not corpus:
            raise ValueError("Korpus tokenizer kosong")
        if min_frequency < 1:
            raise ValueError("min_frequency minimal 1")

        alphabet = sorted({char for sequence in corpus for char in sequence})
        if vocab_size < len(SPECIAL_TOKENS) + len(alphabet):
            raise ValueError("vocab_size terlalu kecil untuk karakter dasar dan token khusus")

        merges: list[tuple[str, str]] = []
        sequences = [sequence[:] for sequence in corpus]
        known_tokens = set(SPECIAL_TOKENS) | set(alphabet)
        while len(known_tokens) < vocab_size:
            counts: Counter[tuple[str, str]] = Counter()
            for sequence in sequences:
                counts.update(
                    pair for pair in zip(sequence, sequence[1:])
                    if pair[0] + pair[1] not in known_tokens
                )
            if not counts:
                break
            best_pair, frequency = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if frequency < min_frequency:
                break
            merges.append(best_pair)
            merged = best_pair[0] + best_pair[1]
            known_tokens.add(merged)
            updated = []
            for sequence in sequences:
                output: list[str] = []
                index = 0
                while index < len(sequence):
                    if (
                        index + 1 < len(sequence)
                        and (sequence[index], sequence[index + 1]) == best_pair
                    ):
                        output.append(merged)
                        index += 2
                    else:
                        output.append(sequence[index])
                        index += 1
                updated.append(output)
            sequences = updated
        return cls(alphabet, merges)

    def encode(self, text: str) -> list[str]:
        tokens = [char if char in self.alphabet else "<unk>" for char in text]
        for pair in self.merges:
            merged = pair[0] + pair[1]
            output: list[str] = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and (tokens[index], tokens[index + 1]) == pair
                ):
                    output.append(merged)
                    index += 2
                else:
                    output.append(tokens[index])
                    index += 1
            tokens = output
        return tokens

    @staticmethod
    def decode(tokens: Iterable[str]) -> str:
        return "".join(token for token in tokens if token not in SPECIAL_TOKENS)

    def to_dict(self) -> dict:
        return {
            "format": "tata-character-bpe",
            "format_version": 1,
            "special_tokens": SPECIAL_TOKENS,
            "alphabet": self.alphabet,
            "merges": [list(pair) for pair in self.merges],
            "vocab_size": len(self.vocab),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CharacterBPE":
        if not isinstance(payload, dict):
            raise ValueError("Payload tokenizer harus berupa objek")
        if payload.get("format") != "tata-character-bpe" or payload.get("format_version") != 1:
            raise ValueError("Format tokenizer tidak dikenal")
        if payload.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("Daftar token khusus tidak cocok")
        alphabet = payload.get("alphabet")
        merges = payload.get("merges")
        if (
            not isinstance(alphabet, list)
            or not all(isinstance(x, str) and len(x) == 1 for x in alphabet)
            or len(set(alphabet)) != len(alphabet)
        ):
            raise ValueError("Alphabet tokenizer tidak valid atau berisi duplikat")

        known_tokens = set(SPECIAL_TOKENS) | set(alphabet)
        validated_merges: list[tuple[str, str]] = []
        if not isinstance(merges, list):
            raise ValueError("Daftar merge tokenizer tidak valid")
        for pair in merges:
            if (
                not isinstance(pair, list) or len(pair) != 2
                or not all(isinstance(part, str) and part for part in pair)
            ):
                raise ValueError("Daftar merge tokenizer tidak valid")
            left, right = pair
            if left not in known_tokens or right not in known_tokens:
                raise ValueError("Merge memakai token yang belum tersedia")
            merged = left + right
            if merged in known_tokens:
                raise ValueError("Merge menghasilkan token duplikat")
            known_tokens.add(merged)
            validated_merges.append((left, right))

        if payload.get("vocab_size") != len(known_tokens):
            raise ValueError("Ukuran vocabulary tokenizer tidak cocok")
        return cls(alphabet, validated_merges)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: Path) -> "CharacterBPE":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_training_texts(jsonl_path: Path, source_register_path: Path) -> list[str]:
    errors, _ = validate_file(jsonl_path)
    if errors:
        raise ValueError("Korpus tidak lolos validasi: " + "; ".join(errors[:5]))
    register_errors, _ = validate_register(source_register_path)
    if register_errors:
        raise ValueError("Register sumber tidak valid: " + "; ".join(register_errors[:5]))
    import csv
    approved = {}
    with source_register_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["decision"].strip().lower() == "approved" and row["training_permission"].strip().lower() == "yes":
                approved[row["source_id"]] = {
                    "title": row["title"].strip(),
                    "uri": row["canonical_uri"].strip(),
                    "license": row["declared_license"].strip().lower(),
                }
    if not approved:
        raise ValueError("Register tidak memiliki sumber approved dengan izin training.")
    texts: list[str] = []
    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        # Hanya contoh yang lolos tinjauan privasi dan kualitas boleh melatih tokenizer.
        if record.get("privacy_review") != "passed" or record.get("quality_review") != "passed":
            continue
        # Unassigned data must be split and reviewed before any tokenizer training.
        if record.get("split") != "train":
            continue
        source = record.get("source")
        if (
            not isinstance(source, dict)
            or not source.get("source_id")
            or source.get("license") not in ALLOWED_LICENSES
        ):
            raise ValueError(f"Sumber/lisensi tidak valid pada baris {line_number}")
        source_id = source["source_id"]
        registered = approved.get(source_id)
        if registered is None:
            raise ValueError(f"source_id {source_id} tidak ada dalam register approved pada baris {line_number}")
        if source.get("title") != registered["title"] or source.get("uri") != registered["uri"]:
            raise ValueError(f"Metadata source tidak cocok dengan register pada baris {line_number}")
        if source.get("license") != registered["license"]:
            raise ValueError(f"Lisensi korpus tidak cocok dengan register pada baris {line_number}")
        messages = record.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Messages tidak valid pada baris {line_number}")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                raise ValueError(f"Role tidak valid pada baris {line_number}")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Content kosong pada baris {line_number}")
            # Sertakan penanda peran agar model dapat mempelajari giliran dialog.
            # Newline is an explicit turn boundary and is also used by generation prompts.
            texts.append(message["role"] + ": " + content + "\n")
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="korpus JSONL yang sudah ditinjau")
    parser.add_argument("--output", type=Path, required=True, help="file tokenizer JSON")
    parser.add_argument("--source-register", type=Path, required=True,
                        help="CSV register sumber dengan izin training yang disetujui")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    texts = load_training_texts(args.corpus, args.source_register)
    tokenizer = CharacterBPE.train(texts, args.vocab_size, args.min_frequency)
    tokenizer.save(args.output)
    print(json.dumps({
        "status": "created",
        "vocab_size": len(tokenizer.vocab),
        "merge_count": len(tokenizer.merges),
        "training_text_count": len(texts),
        "warning": "Eksperimen awal; evaluasi korpus besar berizin masih diperlukan."
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
