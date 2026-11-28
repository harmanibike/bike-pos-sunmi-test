from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.corpus.train_tokenizer import CharacterBPE
from tools.model.training_data import load_reviewed_sequences, tokenizer_vocabulary


REGISTER_FIELDS = [
    "source_id", "title", "owner", "canonical_uri", "accessed_at",
    "declared_license", "license_evidence_uri", "training_permission",
    "commercial_use", "derivative_permission", "attribution_required",
    "redistribution_allowed", "privacy_screen", "reviewer", "decision",
    "decision_notes",
]


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus.jsonl"
        self.tokenizer_path = self.root / "tokenizer.json"
        self.register_path = self.root / "sources.csv"
        self.tokenizer = CharacterBPE.train(["user: halo tata\nassistant: halo\n"], vocab_size=64)
        self.tokenizer.save(self.tokenizer_path)
        self.write_register(["source-train", "source-validation", "source-pending", "source-test"])

    def tearDown(self):
        self.temp.cleanup()

    def write_register(self, source_ids, decision="approved", permission="yes"):
        with self.register_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTER_FIELDS)
            writer.writeheader()
            for source_id in source_ids:
                writer.writerow({
                    "source_id": source_id,
                    "title": "Test",
                    "owner": "Tata",
                    "canonical_uri": "https://example.org/" + source_id,
                    "accessed_at": "2026-11-28",
                    "declared_license": "cc0",
                    "license_evidence_uri": "https://example.org/permission/" + source_id,
                    "training_permission": permission,
                    "commercial_use": "unclear",
                    "derivative_permission": "yes",
                    "attribution_required": "Tidak ada",
                    "redistribution_allowed": "no",
                    "privacy_screen": "Tidak ada data pribadi",
                    "reviewer": "reviewer",
                    "decision": decision,
                    "decision_notes": "Izin training dicatat untuk pengujian.",
                })

    def write_records(self, records):
        self.corpus.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )

    @staticmethod
    def record(example_id, split, quality="passed", source_id=None):
        source_id = source_id or "source-" + example_id
        return {
            "schema_version": 1,
            "example_id": example_id,
            "source": {
                "source_id": source_id, "title": "Test",
                "license": "cc0", "uri": "https://example.org/" + source_id,
            },
            "language": "id",
            "task": "conversation",
            "privacy_review": "passed",
            "quality_review": quality,
            "split": split,
            "messages": [
                {"role": "user", "content": "Halo Tata " + example_id},
                {"role": "assistant", "content": "Jawaban untuk " + example_id},
            ],
        }

    def load(self, split):
        return load_reviewed_sequences(
            self.corpus, self.tokenizer_path, split, self.register_path
        )

    def test_only_requested_approved_split_is_loaded(self):
        self.write_records([
            self.record("train", "train", source_id="source-train"),
            self.record("validation", "validation", source_id="source-validation"),
            self.record("pending", "train", "pending", "source-pending"),
            self.record("test", "test", source_id="source-test"),
        ])
        sequences, metadata = self.load("train")
        self.assertEqual(metadata["records"], 1)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(4, metadata["registered_approved_sources"])
        _, token_to_id = tokenizer_vocabulary(self.tokenizer)
        newline_tokens = self.tokenizer.encode("\n")
        self.assertEqual(1, len(newline_tokens))
        self.assertIn(token_to_id[newline_tokens[0]], sequences[0])

    def test_test_split_is_never_loadable(self):
        with self.assertRaises(ValueError):
            self.load("test")

    def test_tokenizer_mapping_is_deterministic(self):
        first = tokenizer_vocabulary(self.tokenizer)
        second = tokenizer_vocabulary(self.tokenizer)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), len(first[1]))

    def test_invalid_message_is_rejected(self):
        record = self.record("bad", "train", source_id="source-train")
        record["messages"][0]["role"] = "system"
        self.write_records([record])
        with self.assertRaises(ValueError):
            self.load("train")

    def test_unapproved_license_is_rejected(self):
        record = self.record("bad-license", "train", source_id="source-train")
        record["source"]["license"] = "unknown"
        self.write_records([record])
        with self.assertRaises(ValueError):
            self.load("train")

    def test_source_missing_from_register_is_rejected(self):
        self.write_records([self.record("orphan", "train", source_id="not-registered")])
        with self.assertRaisesRegex(ValueError, "tidak ada dalam register"):
            self.load("train")

    def test_pending_source_cannot_enter_training(self):
        self.write_register(["source-pending"], decision="pending")
        self.write_records([self.record("pending", "train", source_id="source-pending")])
        with self.assertRaisesRegex(ValueError, "belum disetujui"):
            self.load("train")

    def test_training_permission_no_or_unclear_is_rejected(self):
        self.write_register(["source-train"], permission="unclear")
        self.write_records([self.record("blocked", "train", source_id="source-train")])
        with self.assertRaisesRegex(ValueError, "training_permission bukan yes"):
            self.load("train")

    def test_source_license_must_match_register(self):
        record = self.record("license-mismatch", "train", source_id="source-train")
        record["source"]["license"] = "cc_by"
        self.write_records([record])
        with self.assertRaisesRegex(ValueError, "Lisensi korpus tidak cocok"):
            self.load("train")

    def test_source_metadata_must_match_register(self):
        record = self.record("mismatch", "train", source_id="source-train")
        record["source"]["uri"] = "https://wrong.example/source"
        self.write_records([record])
        with self.assertRaisesRegex(ValueError, "metadata source"):
            self.load("train")

    def test_empty_register_blocks_training(self):
        self.write_register([])
        self.write_records([self.record("train", "train", source_id="source-train")])
        with self.assertRaisesRegex(ValueError, "Register sumber"):
            self.load("train")


if __name__ == "__main__":
    unittest.main()
