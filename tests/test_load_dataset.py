"""Unit tests for dataset loading and preparation pipeline."""
import sys
import os
import tempfile
import json
import unittest
from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from load_dataset import load_raw_dataset, prepare_dataset


class TestLoadDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_raw_txt(self):
        txt_path = os.path.join(self.temp_dir.name, "sample.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Line one\nLine two\nLine three\n")

        ds = load_raw_dataset(txt_path)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0]["text"], "Line one")

    def test_load_raw_json(self):
        json_path = os.path.join(self.temp_dir.name, "sample.json")
        data = [{"text": "Hello world"}, {"text": "Second row"}]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        ds = load_raw_dataset(json_path)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0]["text"], "Hello world")

    def test_load_unsupported_format(self):
        fake_path = os.path.join(self.temp_dir.name, "sample.xyz")
        with open(fake_path, "w", encoding="utf-8") as f:
            f.write("test")

        with self.assertRaises(ValueError):
            load_raw_dataset(fake_path)

    def test_prepare_dataset_pipeline(self):
        txt_path = os.path.join(self.temp_dir.name, "chat.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("[01/01/24] Messages and calls are end-to-end encrypted.\n")
            f.write("[01/01/24, 09:00] Alice: Reach me at alice@test.org or +254712345678.\n")
            f.write("[01/01/24, 09:05] Bob: We held a clinic day and served 50 patients.\n")

        # Test preprocessing + PII redaction without heavy ML models
        processed = prepare_dataset(
            txt_path,
            clean_text=True,
            classify_relevance=False,
            extract_info=False,
        )

        train_ds = processed
        all_text = " ".join(train_ds["text"])

        # Boilerplate should be cleaned
        self.assertNotIn("end-to-end encrypted", all_text)
        # PII should be redacted
        self.assertNotIn("alice@test.org", all_text)
        self.assertNotIn("+254712345678", all_text)
        self.assertIn("Person0:", all_text)
        self.assertIn("Person1:", all_text)

    def test_standard_columns(self):
        """The text column is renamed (only original_text keeps the raw text) and message_id is added."""
        csv_path = os.path.join(self.temp_dir.name, "msgs.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("transcript\nGrace Wanjiru here. We held a clinic.\nThanks Grace!\n")
        train = prepare_dataset(csv_path)
        self.assertNotIn("transcript", train.column_names)
        self.assertEqual(train["message_id"], [1, 2])
        self.assertEqual(train["original_text"][0], "Grace Wanjiru here. We held a clinic.")
        self.assertNotIn("Grace", " ".join(train["text"]))


if __name__ == "__main__":
    unittest.main()
