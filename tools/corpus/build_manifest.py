#!/usr/bin/env python3
"""Create a privacy-minimal provenance manifest after successful corpus validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from validate_corpus import validate_file
from validate_source_register import validate_register


def create_manifest(corpus_path: Path, source_register_path: Path) -> dict:
    errors, stats = validate_file(corpus_path)
    if errors:
        raise ValueError("Korpus tidak lolos validasi: " + "; ".join(errors[:5]))

    register_errors, _ = validate_register(source_register_path)
    if register_errors:
        raise ValueError("Register sumber tidak valid: " + "; ".join(register_errors[:5]))
    registered = {}
    with source_register_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            registered[row["source_id"]] = row

    raw = corpus_path.read_bytes()
    records = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if any(record.get("privacy_review") != "passed" or record.get("quality_review") != "passed" for record in records):
        raise ValueError("Manifest hanya boleh dibuat jika seluruh contoh lolos tinjauan privasi dan kualitas.")

    source_map: dict[str, dict] = {}
    split_counts: Counter[str] = Counter()

    for record in records:
        source = record["source"]
        source_id = source["source_id"]
        registry_entry = registered.get(source_id)
        if registry_entry is None:
            raise ValueError(f"source_id {source_id} tidak ditemukan dalam register")
        if registry_entry["decision"].strip().lower() != "approved":
            raise ValueError(f"Sumber {source_id} belum disetujui")
        if registry_entry["training_permission"].strip().lower() != "yes":
            raise ValueError(f"Sumber {source_id} tidak memiliki izin training yang jelas")
        if source["title"] != registry_entry["title"].strip() or source.get("uri") != registry_entry["canonical_uri"].strip():
            raise ValueError(f"Metadata sumber tidak cocok dengan register: {source_id}")
        if source_id not in source_map:
            source_map[source_id] = {
                "title": source["title"],
                "license": source["license"],
                "canonical_uri": registry_entry["canonical_uri"].strip(),
                "declared_license": registry_entry["declared_license"].strip(),
                "license_evidence_uri": registry_entry["license_evidence_uri"].strip(),
                "training_permission": registry_entry["training_permission"].strip().lower(),
                "commercial_use": registry_entry["commercial_use"].strip().lower(),
                "example_count": 0,
            }
        elif (
            source_map[source_id]["title"] != source["title"]
            or source_map[source_id]["license"] != source["license"]
        ):
            raise ValueError(f"Metadata sumber tidak konsisten: {source_id}")
        source_map[source_id]["example_count"] += 1
        split_counts[record.get("split", "unassigned")] += 1

    return {
        "manifest_schema_version": 1,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": stats["records"],
        "source_count": len(source_map),
        "sources": [
            {"source_id": key, **source_map[key]}
            for key in sorted(source_map)
        ],
        "split_counts": dict(sorted(split_counts.items())),
        "validation": {
            "status": "passed",
            "validator": "tata-corpus-validator",
            "validator_version": "0.1.0",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="file korpus JSONL")
    parser.add_argument("--output", type=Path, required=True, help="lokasi manifest JSON")
    parser.add_argument("--source-register", type=Path, required=True, help="CSV register sumber yang disetujui")
    args = parser.parse_args()
    try:
        manifest = create_manifest(args.jsonl, args.source_register)
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Gagal membuat manifest: {exc}", file=sys.stderr)
        return 1
    print(f"Manifest berhasil dibuat: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
