import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_manifest import create_manifest


REGISTER_FIELDS = [
    "source_id", "title", "owner", "canonical_uri", "accessed_at",
    "declared_license", "license_evidence_uri", "training_permission",
    "commercial_use", "derivative_permission", "attribution_required",
    "redistribution_allowed", "privacy_screen", "reviewer", "decision",
    "decision_notes",
]


def record(example_id="ex-1", split="train", source_id="source-1"):
    return {
        "schema_version": 1,
        "example_id": example_id,
        "source": {
            "source_id": source_id,
            "title": "Materi milik sendiri",
            "license": "owned_or_explicitly_permitted",
            "uri": "https://example.org/" + source_id,
        },
        "language": "id",
        "task": "instruction_response",
        "messages": [
            {"role": "user", "content": "Apa itu energi?"},
            {"role": "assistant", "content": "Energi adalah kemampuan melakukan usaha."},
        ],
        "privacy_review": "passed",
        "quality_review": "passed",
        "split": split,
    }


class ManifestTests(unittest.TestCase):
    def write_register(self, path, source_ids, decision="approved"):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTER_FIELDS)
            writer.writeheader()
            for source_id in source_ids:
                writer.writerow({
                    "source_id": source_id,
                    "title": "Materi milik sendiri",
                    "owner": "Tim Tata",
                    "canonical_uri": "https://example.org/" + source_id,
                    "accessed_at": "2026-11-28",
                    "declared_license": "Izin eksplisit",
                    "license_evidence_uri": "https://example.org/permission",
                    "training_permission": "yes",
                    "commercial_use": "unclear",
                    "derivative_permission": "yes",
                    "attribution_required": "Atribusi bila dipublikasikan",
                    "redistribution_allowed": "no",
                    "privacy_screen": "Tidak ada data pribadi",
                    "reviewer": "reviewer",
                    "decision": decision,
                    "decision_notes": "Izin training dicatat.",
                })

    def test_manifest_contains_hash_source_provenance_and_split_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            register = Path(directory) / "sources.csv"
            second = record("ex-2", "validation", "source-2")
            second["messages"][0]["content"] = "Bagaimana energi digunakan?"
            second["messages"][1]["content"] = "Energi dapat digunakan untuk melakukan kerja."
            path.write_text(
                json.dumps(record(), ensure_ascii=False) + "\n"
                + json.dumps(second, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.write_register(register, ["source-1", "source-2"])
            manifest = create_manifest(path, register)
        self.assertEqual(2, manifest["record_count"])
        self.assertEqual(2, manifest["source_count"])
        self.assertEqual({"train": 1, "validation": 1}, manifest["split_counts"])
        self.assertEqual("passed", manifest["validation"]["status"])
        self.assertEqual(64, len(manifest["corpus_sha256"]))
        self.assertIn("license_evidence_uri", manifest["sources"][0])

    def test_manifest_refuses_invalid_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            register = Path(directory) / "sources.csv"
            invalid = record()
            invalid["quality_review"] = "pending"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            self.write_register(register, ["source-1"])
            with self.assertRaises(ValueError):
                create_manifest(path, register)

    def test_manifest_refuses_unregistered_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            register = Path(directory) / "sources.csv"
            path.write_text(json.dumps(record()), encoding="utf-8")
            self.write_register(register, ["different-source"])
            with self.assertRaisesRegex(ValueError, "tidak ditemukan dalam register"):
                create_manifest(path, register)

    def test_manifest_refuses_pending_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            register = Path(directory) / "sources.csv"
            path.write_text(json.dumps(record()), encoding="utf-8")
            self.write_register(register, ["source-1"], decision="pending")
            with self.assertRaisesRegex(ValueError, "belum disetujui"):
                create_manifest(path, register)


if __name__ == "__main__":
    unittest.main()
