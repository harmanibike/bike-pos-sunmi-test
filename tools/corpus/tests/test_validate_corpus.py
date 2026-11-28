import json
import tempfile
import unittest
from pathlib import Path

from validate_corpus import validate_file, validate_record


def sample_record():
    return {
        "schema_version": 1,
        "example_id": "contoh-001",
        "source": {
            "source_id": "materi-sendiri-001",
            "title": "Materi milik sendiri",
            "license": "owned_or_explicitly_permitted",
            "uri": None,
        },
        "language": "id",
        "task": "instruction_response",
        "messages": [
            {"role": "user", "content": "Apa itu fotosintesis?"},
            {"role": "assistant", "content": "Proses tumbuhan membuat makanan."},
        ],
        "privacy_review": "passed",
        "quality_review": "pending",
        "split": "unassigned",
    }


class CorpusValidatorTests(unittest.TestCase):
    def test_accepts_valid_record(self):
        self.assertEqual([], validate_record(sample_record(), 1))

    def test_rejects_unknown_license_and_unreviewed_privacy(self):
        record = sample_record()
        record["source"]["license"] = "unknown"
        record["privacy_review"] = "not_reviewed"
        errors = validate_record(record, 1)
        self.assertTrue(any("lisensi" in error for error in errors))
        self.assertTrue(any("privacy_review" in error for error in errors))

    def test_rejects_personal_contact_and_credentials(self):
        record = sample_record()
        record["messages"][0]["content"] = "Hubungi saya 081234567890 atau a@b.com"
        errors = validate_record(record, 1)
        self.assertTrue(any("nomor telepon" in error for error in errors))
        self.assertTrue(any("alamat email" in error for error in errors))

    def test_rejects_non_conversational_roles(self):
        record = sample_record()
        record["messages"][0]["role"] = "system"
        self.assertTrue(any("role" in error for error in validate_record(record, 1)))

    def test_rejects_consecutive_same_speaker(self):
        record = sample_record()
        record["messages"].insert(1, {"role": "user", "content": "Pertanyaan lanjutan"})
        self.assertTrue(any("bergantian" in error for error in validate_record(record, 1)))

    def test_rejects_source_split_leakage(self):
        first = sample_record()
        first["quality_review"] = "passed"
        first["split"] = "train"
        second = sample_record()
        second["example_id"] = "contoh-002"
        second["quality_review"] = "passed"
        second["split"] = "test"
        second["messages"][0]["content"] = "Pertanyaan berbeda"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leakage.jsonl"
            path.write_text(
                json.dumps(first, ensure_ascii=False) + "\n"
                + json.dumps(second, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            errors, _ = validate_file(path)
        self.assertTrue(any("tersebar pada split" in error for error in errors))

    def test_file_detects_duplicate_examples_and_blank_lines(self):
        record = sample_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            line = json.dumps(record, ensure_ascii=False)
            path.write_text(line + "\n" + line + "\n\n", encoding="utf-8")
            errors, stats = validate_file(path)
        self.assertEqual(2, stats["records"])
        self.assertEqual(1, stats["duplicates"])
        self.assertTrue(any("duplikat" in error for error in errors))
        self.assertTrue(any("baris kosong" in error for error in errors))

    def test_rejects_corpus_over_file_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            errors, stats = validate_file(path, max_file_bytes=1)
        self.assertTrue(any("Ukuran korpus" in error for error in errors))
        self.assertEqual(1, stats["limit_blocked"])

    def test_rejects_record_count_over_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "many.jsonl"
            line = json.dumps(sample_record(), ensure_ascii=False)
            path.write_text(line + "\n" + line + "\n", encoding="utf-8")
            errors, stats = validate_file(path, max_records=1)
        self.assertTrue(any("Jumlah record melebihi batas" in error for error in errors))
        self.assertEqual(1, stats["records"])

    def test_rejects_oversized_jsonl_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "line.jsonl"
            path.write_text(json.dumps(sample_record(), ensure_ascii=False) + "\n", encoding="utf-8")
            errors, stats = validate_file(path, max_line_bytes=10)
        self.assertTrue(any("ukuran record melebihi batas" in error for error in errors))
        self.assertEqual(1, stats["limit_blocked"])

    def test_rejects_oversized_message_and_excessive_turns(self):
        record = sample_record()
        record["messages"][0]["content"] = "x" * 20_001
        record["messages"].extend(
            [{"role": "user", "content": "lanjutan"}, {"role": "assistant", "content": "jawaban"}]
            * 20
        )
        errors = validate_record(record, 1)
        self.assertTrue(any("melebihi batas 20000 karakter" in error for error in errors))
        self.assertTrue(any("jumlah pesan melebihi batas 40" in error for error in errors))

    def test_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text("{not json}\n", encoding="utf-8")
            errors, _ = validate_file(path)
        self.assertTrue(any("JSON tidak valid" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
