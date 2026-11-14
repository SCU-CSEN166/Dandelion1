"""Unit tests for the preprocessing stage."""
import sys
import os
import unittest
from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import clean_chat_text, preprocess_dataset


class TestPreprocess(unittest.TestCase):

    def test_clean_chat_text_encryption_notice(self):
        text = "[10/12/23, 08:30:15] Messages and calls are end-to-end encrypted. No one outside of this chat can read them.\nHello team!"
        cleaned = clean_chat_text(text)
        self.assertNotIn("encrypted", cleaned)
        self.assertIn("Hello team!", cleaned)

    def test_clean_chat_text_standalone_encryption(self):
        text = "Messages and calls are end-to-end encrypted.\nGood morning everyone."
        cleaned = clean_chat_text(text)
        self.assertNotIn("encrypted", cleaned)
        self.assertIn("Good morning everyone.", cleaned)

    def test_clean_chat_text_media_omitted(self):
        text = "[10/12/23, 09:15:00] Alice: image omitted\nLet's check the clinic reports."
        cleaned = clean_chat_text(text)
        self.assertNotIn("image omitted", cleaned)
        self.assertIn("Let's check the clinic reports.", cleaned)

    def test_clean_chat_text_deleted_message(self):
        text = "[10/12/23, 09:20:00] Bob: This message was deleted\nMeeting at 2pm."
        cleaned = clean_chat_text(text)
        self.assertNotIn("This message was deleted", cleaned)
        self.assertIn("Meeting at 2pm.", cleaned)

    def test_clean_chat_text_attached_and_contact(self):
        text = "[10/12/23, 10:00:00] Charlie: <attached: report.pdf>\nSee attached file."
        cleaned = clean_chat_text(text)
        self.assertNotIn("<attached:", cleaned)
        self.assertIn("See attached file.", cleaned)

    def test_clean_chat_text_empty_and_whitespace(self):
        self.assertEqual(clean_chat_text(""), "")
        self.assertEqual(clean_chat_text(None), "")
        self.assertEqual(clean_chat_text("   \n\n  "), "")

    def test_preprocess_dataset(self):
        raw_rows = [
            {"text": "Messages and calls are end-to-end encrypted.\nWe reached 30 women today."},
            {"text": "[10/12/23, 11:00:00] Grace: image omitted\nAll supplies delivered."},
            {"text": "Regular message with no boilerplate."},
        ]
        ds = Dataset.from_list(raw_rows)
        processed = preprocess_dataset(ds, text_col="text")

        self.assertNotIn("encrypted", processed[0]["text"])
        self.assertIn("We reached 30 women today.", processed[0]["text"])
        self.assertNotIn("image omitted", processed[1]["text"])
        self.assertIn("All supplies delivered.", processed[1]["text"])
        self.assertEqual(processed[2]["text"], "Regular message with no boilerplate.")


if __name__ == "__main__":
    unittest.main()
