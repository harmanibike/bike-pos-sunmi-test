#!/usr/bin/env python3
"""Validate Tata AI JSONL training examples without uploading or modifying data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ALLOWED_LICENSES = {
    "public_domain", "cc0", "cc_by", "cc_by_sa",
    "owned_or_explicitly_permitted", "explicit_permission",
}
ALLOWED_ROLES = {"user", "assistant"}
ALLOWED_SPLITS = {"unassigned", "train", "validation", "test"}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?62|0)\s?8\d[\d\s-]{7,13}(?!\w)")
SECRET_RE = re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*\S{8,}")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")

# Hard ceilings for untrusted corpus intake. These are guardrails, not training targets.
MAX_CORPUS_BYTES = 100 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_LINE_BYTES = 1 * 1024 * 1024
MAX_MESSAGES_PER_RECORD = 40
MAX_CONTENT_CHARS = 20_000
MAX_TOTAL_CONTENT_CHARS = 100_000


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def fingerprint(messages: list[dict[str, str]]) -> str:
    canonical = "\n".join(
        f"{item['role']}:{normalize_text(item['content'])}" for item in messages
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_record(record: Any, line_number: int) -> list[str]:
    errors: list[str] = []
    prefix = f"baris {line_number}"
    if not isinstance(record, dict):
        return [f"{prefix}: setiap baris harus berupa objek JSON"]

    if record.get("schema_version") != 1:
        errors.append(f"{prefix}: schema_version harus 1")
    for field in ("example_id", "language", "task"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            errors.append(f"{prefix}: {field} wajib berupa teks tidak kosong")

    source = record.get("source")
    if not isinstance(source, dict):
        errors.append(f"{prefix}: source wajib berupa objek")
    else:
        for field in ("source_id", "title", "license"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                errors.append(f"{prefix}: source.{field} wajib diisi")
        if source.get("license") not in ALLOWED_LICENSES:
            errors.append(f"{prefix}: lisensi sumber tidak ada dalam daftar yang diizinkan")
        uri = source.get("uri")
        if uri is not None and not isinstance(uri, str):
            errors.append(f"{prefix}: source.uri harus teks atau null")

    if record.get("privacy_review") not in {"pending", "passed"}:
        errors.append(f"{prefix}: privacy_review harus 'pending' atau 'passed'")
    if record.get("quality_review") not in {"pending", "passed", "rejected"}:
        errors.append(f"{prefix}: quality_review harus 'pending', 'passed', atau 'rejected'")
    if record.get("split", "unassigned") not in ALLOWED_SPLITS:
        errors.append(f"{prefix}: split tidak dikenal")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append(f"{prefix}: messages harus berisi minimal dua pesan")
        return errors

    if len(messages) > MAX_MESSAGES_PER_RECORD:
        errors.append(f"{prefix}: jumlah pesan melebihi batas {MAX_MESSAGES_PER_RECORD}")
    valid_messages: list[dict[str, str]] = []
    total_content_chars = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"{prefix}: messages[{index}] harus berupa objek")
            continue
        role, content = message.get("role"), message.get("content")
        if role not in ALLOWED_ROLES:
            errors.append(f"{prefix}: role messages[{index}] tidak diizinkan")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{prefix}: content messages[{index}] wajib diisi")
            continue
        total_content_chars += len(content)
        if len(content) > MAX_CONTENT_CHARS:
            errors.append(f"{prefix}: content messages[{index}] melebihi batas {MAX_CONTENT_CHARS} karakter")
        if role in ALLOWED_ROLES:
            valid_messages.append({"role": role, "content": content})
        for pattern, label in (
            (EMAIL_RE, "alamat email"),
            (PHONE_RE, "nomor telepon"),
            (SECRET_RE, "kemungkinan kredensial"),
            (PRIVATE_KEY_RE, "private key"),
        ):
            if pattern.search(content):
                errors.append(f"{prefix}: terdeteksi {label} pada messages[{index}]")

    if total_content_chars > MAX_TOTAL_CONTENT_CHARS:
        errors.append(f"{prefix}: total teks percakapan melebihi batas {MAX_TOTAL_CONTENT_CHARS} karakter")
    if valid_messages and valid_messages[0]["role"] != "user":
        errors.append(f"{prefix}: pesan pertama harus dari user")
    if valid_messages and valid_messages[-1]["role"] != "assistant":
        errors.append(f"{prefix}: pesan terakhir harus dari assistant")
    for index in range(1, len(valid_messages)):
        if valid_messages[index]["role"] == valid_messages[index - 1]["role"]:
            errors.append(f"{prefix}: giliran pesan harus bergantian user dan assistant")
            break
    return errors


def validate_file(
    path: Path,
    max_file_bytes: int = MAX_CORPUS_BYTES,
    max_records: int = MAX_RECORDS,
    max_line_bytes: int = MAX_LINE_BYTES,
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    stats = {"records": 0, "duplicates": 0, "privacy_blocked": 0, "limit_blocked": 0}
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    source_splits: dict[str, str] = {}

    try:
        file_size = path.stat().st_size
        if file_size > max_file_bytes:
            stats["limit_blocked"] += 1
            return [f"Ukuran korpus {file_size} byte melebihi batas {max_file_bytes} byte"], stats
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"Gagal membaca file UTF-8: {exc}"], stats

    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if line_number > max_records:
            stats["limit_blocked"] += 1
            errors.append(f"Jumlah record melebihi batas {max_records}")
            break
        if len(raw_line.encode("utf-8")) > max_line_bytes:
            stats["limit_blocked"] += 1
            errors.append(f"baris {line_number}: ukuran record melebihi batas {max_line_bytes} byte")
            continue
        if not raw_line.strip():
            errors.append(f"baris {line_number}: baris kosong tidak diizinkan")
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(f"baris {line_number}: JSON tidak valid ({exc.msg})")
            continue

        stats["records"] += 1
        record_errors = validate_record(record, line_number)
        errors.extend(record_errors)
        if any("terdeteksi " in item for item in record_errors):
            stats["privacy_blocked"] += 1

        if isinstance(record, dict):
            source = record.get("source")
            split = record.get("split", "unassigned")
            if (
                record.get("privacy_review") == "passed"
                and record.get("quality_review") == "passed"
                and isinstance(source, dict)
                and isinstance(source.get("source_id"), str)
                and split in {"train", "validation", "test"}
            ):
                source_id = source["source_id"]
                previous_split = source_splits.get(source_id)
                if previous_split is not None and previous_split != split:
                    errors.append(
                        f"baris {line_number}: source_id yang sama tersebar pada "
                        f"split {previous_split} dan {split}"
                    )
                else:
                    source_splits[source_id] = split

            example_id = record.get("example_id")
            if isinstance(example_id, str) and example_id:
                if example_id in seen_ids:
                    errors.append(f"baris {line_number}: example_id duplikat")
                seen_ids.add(example_id)

            messages = record.get("messages")
            if isinstance(messages, list) and all(
                isinstance(m, dict) and isinstance(m.get("role"), str)
                and isinstance(m.get("content"), str) for m in messages
            ):
                key = fingerprint(messages)
                if key in seen_fingerprints:
                    stats["duplicates"] += 1
                    errors.append(f"baris {line_number}: contoh percakapan duplikat")
                seen_fingerprints.add(key)

    return errors, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="file korpus JSONL UTF-8")
    args = parser.parse_args()
    errors, stats = validate_file(args.jsonl)
    print(json.dumps({"status": "failed" if errors else "passed", **stats,
                      "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
