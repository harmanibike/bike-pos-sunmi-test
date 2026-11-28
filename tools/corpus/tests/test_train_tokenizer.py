import tempfile
import unittest
from pathlib import Path

import json
import csv

from train_tokenizer import CharacterBPE, SPECIAL_TOKENS, load_training_texts


class CharacterBPETests(unittest.TestCase):
    def write_register(self, path, source_ids):
        fields = ["source_id", "title", "owner", "canonical_uri", "accessed_at",
                  "declared_license", "license_evidence_uri", "training_permission",
                  "commercial_use", "derivative_permission", "attribution_required",
                  "redistribution_allowed", "privacy_screen", "reviewer", "decision",
                  "decision_notes"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for source_id in source_ids:
                writer.writerow({
                    "source_id": source_id, "title": "Sumber uji", "owner": "Tata",
                    "canonical_uri": "https://example.org/" + source_id,
                    "accessed_at": "2026-11-28", "declared_license": "cc0",
                    "license_evidence_uri": "https://example.org/license",
                    "training_permission": "yes", "commercial_use": "unclear",
                    "derivative_permission": "yes", "attribution_required": "Tidak ada",
                    "redistribution_allowed": "no", "privacy_screen": "Aman",
                    "reviewer": "reviewer", "decision": "approved",
                    "decision_notes": "Izin training eksplisit."
                })
    def test_shared_python_kotlin_parity_fixture(self):
        repository_root = Path(__file__).resolve().parents[3]
        fixture_path = repository_root / "app/src/test/resources/character_bpe_parity.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        tokenizer = CharacterBPE.from_dict(fixture["tokenizer"])
        ordered_tokens = list(SPECIAL_TOKENS) + list(tokenizer.alphabet)
        ordered_tokens.extend(left + right for left, right in tokenizer.merges)
        token_to_id = {token: index for index, token in enumerate(ordered_tokens)}
        cases = [{"tokenizer": fixture["tokenizer"], "samples": fixture["samples"]}]
        cases.extend(fixture.get("additional_cases", []))
        for case in cases:
            case_tokenizer = CharacterBPE.from_dict(case["tokenizer"])
            case_vocab = list(SPECIAL_TOKENS) + list(case_tokenizer.alphabet)
            case_vocab.extend(left + right for left, right in case_tokenizer.merges)
            case_token_to_id = {token: index for index, token in enumerate(case_vocab)}
            for sample in case["samples"]:
                with self.subTest(text=sample["text"]):
                    tokens = case_tokenizer.encode(sample["text"])
                    self.assertEqual(sample["token_ids"], [case_token_to_id[token] for token in tokens])
                    self.assertEqual(sample["decoded"], case_tokenizer.decode(tokens))

    def test_round_trip_indonesian_text(self):
        texts = [
            "Tata belajar Bahasa Indonesia.",
            "Belajar bahasa Indonesia bersama Tata.",
            "Angka 10.000 dan kata berimbuhan membaca."
        ]
        tokenizer = CharacterBPE.train(texts, vocab_size=128, min_frequency=2)
        sample = "Tata belajar Bahasa Indonesia."
        self.assertEqual(sample, tokenizer.decode(tokenizer.encode(sample)))

    def test_training_is_deterministic(self):
        texts = ["membaca buku", "membaca cerita", "buku cerita"]
        first = CharacterBPE.train(texts, vocab_size=64, min_frequency=2)
        second = CharacterBPE.train(texts, vocab_size=64, min_frequency=2)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_unknown_character_uses_unknown_token(self):
        tokenizer = CharacterBPE.train(["tata ai"], vocab_size=32)
        self.assertIn("<unk>", tokenizer.encode("tata 🧩"))

    def test_save_and_load_preserves_encoding(self):
        tokenizer = CharacterBPE.train(["kata-kata sederhana", "kata yang sederhana"], vocab_size=64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = CharacterBPE.load(path)
        self.assertEqual(tokenizer.encode("kata sederhana"), loaded.encode("kata sederhana"))
        self.assertEqual(tokenizer.to_dict(), loaded.to_dict())

    def test_rejects_too_small_vocabulary(self):
        with self.assertRaises(ValueError):
            CharacterBPE.train(["Indonesia"], vocab_size=4)

    def test_merge_vocabulary_contains_only_unique_tokens(self):
        tokenizer = CharacterBPE.train(
            ["kata kata kata", "kata-kata kata"], vocab_size=64, min_frequency=1
        )
        ordered_tokens = list(SPECIAL_TOKENS) + list(tokenizer.alphabet)
        ordered_tokens.extend(left + right for left, right in tokenizer.merges)
        self.assertEqual(len(tokenizer.vocab), len(ordered_tokens))
        self.assertEqual(len(ordered_tokens), len(set(ordered_tokens)))
        payload = tokenizer.to_dict()
        restored = CharacterBPE.from_dict(payload)
        self.assertEqual(tokenizer.to_dict(), restored.to_dict())

    def test_rejects_duplicate_alphabet_entries(self):
        payload = CharacterBPE.train(["tata"], vocab_size=32).to_dict()
        payload["alphabet"].append(payload["alphabet"][0])
        with self.assertRaises(ValueError):
            CharacterBPE.from_dict(payload)

    def test_rejects_merge_using_unknown_symbol(self):
        payload = CharacterBPE.train(["tata"], vocab_size=32).to_dict()
        payload["merges"] = [["tidak-ada", "t"]]
        payload["vocab_size"] = len(set(payload["special_tokens"] + payload["alphabet"])) + 1
        with self.assertRaises(ValueError):
            CharacterBPE.from_dict(payload)

    def test_rejects_incorrect_declared_vocabulary_size(self):
        payload = CharacterBPE.train(["tata"], vocab_size=32).to_dict()
        payload["vocab_size"] += 1
        with self.assertRaises(ValueError):
            CharacterBPE.from_dict(payload)

    def test_rejects_two_merge_paths_with_same_token(self):
        payload = {
            "format": "tata-character-bpe",
            "format_version": 1,
            "special_tokens": SPECIAL_TOKENS,
            "alphabet": ["a", "b", "c"],
            "merges": [["a", "b"], ["ab", "c"], ["b", "c"], ["a", "bc"]],
            "vocab_size": len(SPECIAL_TOKENS) + 3 + 4,
        }
        with self.assertRaises(ValueError):
            CharacterBPE.from_dict(payload)

    def test_training_loader_excludes_unreviewed_and_held_out_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            register = Path(directory) / "sources.csv"
            self.write_register(register, ["sumber-1", "sumber-2", "sumber-3", "sumber-4"])
            base = {
                "schema_version": 1,
                "example_id": "latih-1",
                "language": "id",
                "task": "conversation",
                "privacy_review": "passed",
                "quality_review": "passed",
                "split": "train",
                "source": {"source_id": "sumber-1", "title": "Sumber uji", "license": "cc0", "uri": "https://example.org/sumber-1"},
                "messages": [
                    {"role": "user", "content": "teks latih"},
                    {"role": "assistant", "content": "jawaban latih"},
                ],
            }
            pending = {**base, "example_id": "pending-1", "quality_review": "pending",
                       "source": {"source_id": "sumber-2", "title": "Sumber uji", "license": "cc0", "uri": "https://example.org/sumber-2"},
                       "messages": [{"role": "user", "content": "teks pending"},
                                    {"role": "assistant", "content": "jawaban pending"}]}
            held_out = {**base, "example_id": "test-1", "split": "test",
                        "source": {"source_id": "sumber-3", "title": "Sumber uji", "license": "cc0", "uri": "https://example.org/sumber-3"},
                        "messages": [{"role": "user", "content": "teks test"},
                                     {"role": "assistant", "content": "jawaban test"}]}
            unassigned = {**base, "example_id": "unassigned-1", "split": "unassigned",
                          "source": {"source_id": "sumber-4", "title": "Sumber uji", "license": "cc0", "uri": "https://example.org/sumber-4"},
                          "messages": [{"role": "user", "content": "teks belum dibagi"},
                                       {"role": "assistant", "content": "jawaban belum dibagi"}]}
            path.write_text(
                "\n".join(json.dumps(item) for item in [base, pending, held_out, unassigned]),
                encoding="utf-8",
            )
            self.assertEqual(
                ["user: teks latih\n", "assistant: jawaban latih\n"],
                load_training_texts(path, register),
            )

    def test_training_loader_rejects_license_mismatch_with_register(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            register = Path(directory) / "sources.csv"
            self.write_register(register, ["sumber"])
            record = {
                "schema_version": 1, "example_id": "license-mismatch",
                "language": "id", "task": "conversation",
                "privacy_review": "passed", "quality_review": "passed", "split": "train",
                "source": {"source_id": "sumber", "title": "Sumber uji", "license": "cc_by", "uri": "https://example.org/sumber"},
                "messages": [{"role": "user", "content": "pertanyaan"}, {"role": "assistant", "content": "jawaban"}],
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Lisensi korpus tidak cocok"):
                load_training_texts(path, register)

    def test_training_loader_rejects_unapproved_license(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            register = Path(directory) / "sources.csv"
            self.write_register(register, ["sumber"])
            record = {
                "schema_version": 1,
                "example_id": "lisensi-buruk",
                "language": "id",
                "task": "conversation",
                "privacy_review": "passed",
                "quality_review": "passed",
                "split": "train",
                "source": {"source_id": "sumber", "title": "Sumber uji", "license": "unknown", "uri": "https://example.org/sumber"},
                "messages": [
                    {"role": "user", "content": "pertanyaan"},
                    {"role": "assistant", "content": "jawaban"},
                ],
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_training_texts(path, register)


if __name__ == "__main__":
    unittest.main()
