"""Load only reviewed train/validation examples backed by approved source permissions.

The test split is deliberately never loaded by this module.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from tools.corpus.train_tokenizer import CharacterBPE, SPECIAL_TOKENS
    from tools.corpus.validate_corpus import validate_file
    from tools.corpus.validate_source_register import validate_register
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
    from train_tokenizer import CharacterBPE, SPECIAL_TOKENS
    from validate_corpus import validate_file
    from validate_source_register import validate_register


def tokenizer_vocabulary(tokenizer: CharacterBPE) -> tuple[list[str], dict[str, int]]:
    ordered = list(SPECIAL_TOKENS)
    ordered.extend(tokenizer.alphabet)
    for left, right in tokenizer.merges:
        merged = left + right
        if merged not in ordered:
            ordered.append(merged)
    return ordered, {token: index for index, token in enumerate(ordered)}


def load_approved_source_register(register_path: Path) -> dict[str, dict[str, str]]:
    errors, _ = validate_register(register_path)
    if errors:
        raise ValueError("Register sumber tidak valid: " + "; ".join(errors[:5]))
    approved: dict[str, dict[str, str]] = {}
    with register_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            approved[row["source_id"]] = {
                "title": row["title"].strip(),
                "canonical_uri": row["canonical_uri"].strip(),
                "declared_license": row["declared_license"].strip(),
                "license_evidence_uri": row["license_evidence_uri"].strip(),
                "training_permission": row["training_permission"].strip().lower(),
                "commercial_use": row["commercial_use"].strip().lower(),
                "decision": row["decision"].strip().lower(),
            }
    return approved


def load_reviewed_sequences(
    corpus_path: Path,
    tokenizer_path: Path,
    split: str,
    source_register_path: Path,
) -> tuple[list[list[int]], dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise ValueError("Trainer hanya boleh memuat split train atau validation")
    errors, _ = validate_file(corpus_path)
    if errors:
        raise ValueError("Korpus tidak lolos validasi: " + "; ".join(errors[:5]))
    approved_sources = load_approved_source_register(source_register_path)
    tokenizer = CharacterBPE.load(tokenizer_path)
    vocabulary, token_to_id = tokenizer_vocabulary(tokenizer)
    sequences: list[list[int]] = []
    source_ids: set[str] = set()
    record_count = 0
    unknown_id = token_to_id["<unk>"]
    bos_id = token_to_id["<bos>"]
    eos_id = token_to_id["<eos>"]

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"Baris kosong tidak diizinkan: {line_number}")
            record = json.loads(raw)
            if record.get("privacy_review") != "passed" or record.get("quality_review") != "passed":
                continue
            if record.get("split") != split:
                continue
            source = record.get("source")
            if not isinstance(source, dict) or not source.get("source_id"):
                raise ValueError(f"source tidak valid pada baris {line_number}")
            source_id = source["source_id"]
            registered = approved_sources.get(source_id)
            if registered is None:
                raise ValueError(
                    f"source_id {source_id} tidak ada dalam register approved pada baris {line_number}"
                )
            if registered["decision"] != "approved":
                raise ValueError(f"Sumber {source_id} belum disetujui untuk training")
            if registered["training_permission"] != "yes":
                raise ValueError(f"Sumber {source_id}: training_permission bukan yes")
            if source.get("title") != registered["title"] or source.get("uri") != registered["canonical_uri"]:
                raise ValueError(
                    f"metadata source (title/uri) tidak cocok dengan register pada baris {line_number}"
                )
            allowed_licenses = {
                "public_domain", "cc0", "cc_by", "cc_by_sa",
                "owned_or_explicitly_permitted", "explicit_permission",
            }
            if source.get("license") not in allowed_licenses:
                raise ValueError(f"Lisensi korpus tidak dikenal pada baris {line_number}")
            if source.get("license") != registered["declared_license"].strip().lower():
                raise ValueError(f"Lisensi korpus tidak cocok dengan register pada baris {line_number}")
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"messages tidak valid pada baris {line_number}")
            ids = [bos_id]
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                    raise ValueError(f"role tidak valid pada baris {line_number}")
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"content kosong pada baris {line_number}")
                role_tokens = tokenizer.encode(message["role"] + ": ")
                ids.extend(token_to_id.get(token, unknown_id) for token in role_tokens)
                ids.extend(token_to_id.get(token, unknown_id) for token in tokenizer.encode(content))
                ids.extend(token_to_id.get(token, unknown_id) for token in tokenizer.encode("\n"))
            ids.append(eos_id)
            if len(ids) >= 3:
                sequences.append(ids)
                source_ids.add(source_id)
                record_count += 1

    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    tokenizer_digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    register_digest = hashlib.sha256(source_register_path.read_bytes()).hexdigest()
    metadata = {
        "split": split,
        "records": record_count,
        "source_groups": len(source_ids),
        "registered_approved_sources": sum(1 for item in approved_sources.values() if item["decision"] == "approved" and item["training_permission"] == "yes"),
        "corpus_sha256": digest,
        "tokenizer_sha256": tokenizer_digest,
        "source_register_sha256": register_digest,
        "source_ids": sorted(source_ids),
    }
    return sequences, metadata
