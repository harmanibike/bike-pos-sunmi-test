import random
import sys
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_transformer
from train_transformer import best_checkpoint_path, early_stop_reached, load_resume_checkpoint, atomic_save
from unittest.mock import Mock, patch


class BestCheckpointPathTests(unittest.TestCase):
    def test_best_checkpoint_uses_separate_suffix(self):
        output = Path("runs/tata-v0.pt")
        self.assertEqual(Path("runs/tata-v0.best.pt"), best_checkpoint_path(output))

    def test_best_checkpoint_keeps_non_pt_suffix(self):
        output = Path("model.checkpoint")
        self.assertEqual(Path("model.best.checkpoint"), best_checkpoint_path(output))

    def test_early_stopping_disabled_with_zero_patience(self):
        self.assertFalse(early_stop_reached(0, 100))

    def test_early_stopping_waits_until_patience(self):
        self.assertFalse(early_stop_reached(3, 2))
        self.assertTrue(early_stop_reached(3, 3))
        self.assertTrue(early_stop_reached(3, 4))



class TrainingResumeIntegrationTests(unittest.TestCase):
    def test_training_checkpoint_can_resume_and_continue_steps(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch tidak tersedia pada environment ini")
        train_meta = {"vocab_size": 8, "records": 2, "corpus_sha256": "train-hash",
                      "tokenizer_sha256": "tokenizer-hash", "source_register_sha256": "register-hash"}
        validation_meta = {"vocab_size": 8, "records": 2, "corpus_sha256": "validation-hash",
                           "source_register_sha256": "register-hash"}
        def fake_load(corpus, tokenizer, split, register):
            return [[1, 2, 3], [2, 3, 4], [3, 4, 5]], train_meta if split == "train" else validation_meta
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "resume.pt"
            common = ["--corpus", str(root / "corpus.jsonl"), "--tokenizer", str(root / "tokenizer.json"),
                      "--source-register", str(root / "sources.csv"), "--output", str(output),
                      "--steps", "2", "--batch-size", "1", "--context-length", "2",
                      "--eval-interval", "1", "--checkpoint-interval", "1",
                      "--early-stopping-patience", "0", "--device", "cpu"]
            with patch.object(train_transformer, "load_reviewed_sequences", side_effect=fake_load):
                with patch.object(sys, "argv", ["train_transformer.py", *common]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(train_transformer.main(), 0)
                first = load_resume_checkpoint(torch, output, "cpu")
                self.assertEqual(first["step"], 2)
                with patch.object(sys, "argv", ["train_transformer.py", *common, "--resume"]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(train_transformer.main(), 0)
            resumed = load_resume_checkpoint(torch, output, "cpu")
            self.assertEqual(resumed["step"], 4)
            self.assertTrue(train_transformer.best_checkpoint_path(output).exists())
            self.assertEqual(resumed["train_meta"]["corpus_sha256"], "train-hash")
            self.assertEqual(resumed["validation_meta"]["corpus_sha256"], "validation-hash")


class SafeCheckpointLoadingTests(unittest.TestCase):
    def valid_checkpoint(self):
        return {
            "format": "tata-transformer-checkpoint",
            "format_version": 1,
            "checkpoint_role": "latest_resume",
            "step": 12,
            "config": {},
            "model": {},
            "optimizer": {},
            "train_meta": {},
            "validation_meta": {},
        }

    def test_load_uses_weights_only_and_validates_checkpoint(self):
        torch = Mock()
        torch.load.return_value = self.valid_checkpoint()
        result = load_resume_checkpoint(torch, Path("resume.pt"), "cpu")
        torch.load.assert_called_once_with(
            Path("resume.pt"), map_location="cpu", weights_only=True
        )
        self.assertEqual(result["step"], 12)

    def test_rejects_wrong_checkpoint_format(self):
        torch = Mock()
        torch.load.return_value = {"format": "other"}
        with self.assertRaisesRegex(ValueError, "bukan milik Tata"):
            load_resume_checkpoint(torch, Path("bad.pt"), "cpu")

    def test_rejects_incomplete_checkpoint(self):
        torch = Mock()
        checkpoint = self.valid_checkpoint()
        del checkpoint["optimizer"]
        torch.load.return_value = checkpoint
        with self.assertRaisesRegex(ValueError, "tidak lengkap"):
            load_resume_checkpoint(torch, Path("bad.pt"), "cpu")


    def test_real_checkpoint_round_trip_with_rng_state_is_weights_only_safe(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch tidak tersedia pada environment ini")
        payload = self.valid_checkpoint()
        payload["model"] = {"weight": torch.tensor([1.0, 2.0])}
        payload["optimizer"] = {"state": {}, "param_groups": []}
        payload["python_rng_state"] = random.Random(7).getstate()
        payload["torch_rng_state"] = torch.get_rng_state()
        payload["cuda_rng_state"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            atomic_save(torch, payload, path)
            loaded = load_resume_checkpoint(torch, path, "cpu")
        self.assertEqual(loaded["step"], 12)
        self.assertTrue(torch.equal(loaded["model"]["weight"], payload["model"]["weight"]))
        self.assertEqual(loaded["python_rng_state"], payload["python_rng_state"])

    def test_does_not_fall_back_when_weights_only_is_unsupported(self):
        torch = Mock()
        torch.load.side_effect = TypeError("unsupported keyword")
        with self.assertRaisesRegex(RuntimeError, "pemuatan checkpoint aman"):
            load_resume_checkpoint(torch, Path("legacy.pt"), "cpu")
        torch.load.assert_called_once()



if __name__ == "__main__":
    unittest.main()
