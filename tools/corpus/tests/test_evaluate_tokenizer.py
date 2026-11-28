import json
import tempfile
import unittest
from pathlib import Path

from evaluate_tokenizer import DEFAULT_PROBES, atomic_write_json, build_report, evaluate_text


class TokenizerEvaluationTests(unittest.TestCase):
    def setUp(self):
        from train_tokenizer import CharacterBPE
        self.temp = tempfile.TemporaryDirectory()
        self.tokenizer_path = Path(self.temp.name) / "tokenizer.json"
        tokenizer = CharacterBPE.train(
            [" ".join(DEFAULT_PROBES[:-1]) + " emoji uji"], vocab_size=256, min_frequency=1
        )
        tokenizer.save(self.tokenizer_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_probe_covers_indonesian_edges(self):
        joined = " ".join(DEFAULT_PROBES).lower()
        for expected in ("membacakan", "kata-kata", "rp10.000,50", "offline-first", "kediri"):
            self.assertIn(expected, joined)

    def test_known_text_round_trips(self):
        from train_tokenizer import CharacterBPE
        tokenizer = CharacterBPE.load(self.tokenizer_path)
        result = evaluate_text(tokenizer, "kata-kata")
        self.assertTrue(result["round_trip_exact"])
        self.assertEqual(0, result["unknown_tokens"])

    def test_unknown_characters_are_counted(self):
        from train_tokenizer import CharacterBPE
        tokenizer = CharacterBPE.train(["Tata AI"], vocab_size=32)
        result = evaluate_text(tokenizer, "Tata 😀")
        self.assertGreater(result["unknown_tokens"], 0)
        self.assertFalse(result["round_trip_exact"])

    def test_report_has_hash_and_aggregate_metrics(self):
        report = build_report(self.tokenizer_path, ["Rp10.000", "membaca"])
        self.assertEqual(2, report["probe_count"])
        self.assertEqual(2, report["summary"]["exact_round_trip_count"])
        self.assertEqual(64, len(report["tokenizer_sha256"]))
        self.assertEqual(0, report["summary"]["unknown_tokens"])

    def test_report_does_not_store_custom_probe_text(self):
        secret_probe = "TEKS-RAHASIA-UNTUK-UJI-123"
        report = build_report(self.tokenizer_path, [secret_probe])
        rendered = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(secret_probe, rendered)
        self.assertEqual("probe-001", report["probes"][0]["probe_id"])
        self.assertNotIn("text", report["probes"][0])

    def test_report_rejects_empty_custom_probe(self):
        with self.assertRaises(ValueError):
            build_report(self.tokenizer_path, [""])

    def test_atomic_report_is_valid_json(self):
        report = build_report(self.tokenizer_path, ["kata-kata"])
        output = Path(self.temp.name) / "reports" / "tokenizer.json"
        atomic_write_json(output, report)
        loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report, loaded)
        self.assertEqual([], list(output.parent.glob("*.tmp")))

if __name__ == "__main__":
    unittest.main()
