"""Unit tests for the PII redaction stage."""
import sys
import os
import unittest
from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from redact import redact_line, scan_batch, process_dataset_pii


def redact_text(text, namemap, bare_senders=False):
    return redact_line(text, namemap, bare_senders)[0]


class TestRedact(unittest.TestCase):

    def test_redact_email(self):
        namemap = {"map": {}, "count": 0}
        text = "Contact me at user.name@example.org or info@dandelionafrica.org please."
        redacted = redact_text(text, namemap)
        self.assertEqual(redacted, "Contact me at [EMAIL] or [EMAIL] please.")

    def test_redact_phone_international(self):
        namemap = {"map": {}, "count": 0}
        text = "Call +1 (555) 234-5678 or 555-876-5432 today."
        redacted = redact_text(text, namemap)
        self.assertNotIn("555) 234-5678", redacted)
        self.assertNotIn("555-876-5432", redacted)
        self.assertEqual(redacted.count("[PHONE]"), 2)

    def test_redact_phone_kenyan(self):
        namemap = {"map": {}, "count": 0}
        text = "Reach Sister Nancy at +254 712 345 678 or 07560080089."
        redacted = redact_text(text, namemap)
        self.assertNotIn("254 712 345 678", redacted)
        self.assertNotIn("07560080089", redacted)
        self.assertEqual(redacted.count("[PHONE]"), 2)

    def test_sender_anonymization_and_body_mentions(self):
        namemap = {"map": {}, "count": 0}
        line1 = "[12/03/24, 10:15:22] Grace Wanjiru: We served 40 women at the dispensary today."
        red1 = redact_text(line1, namemap)
        self.assertTrue(red1.startswith("[12/03/24, 10:15:22] Person0: "))
        self.assertNotIn("Grace Wanjiru", red1)

        line2 = "12/03/2024, 10:16 - John Doe: I spoke with Grace Wanjiru earlier about the medicine."
        red2 = redact_text(line2, namemap)
        self.assertTrue(red2.startswith("12/03/2024, 10:16 - Person1: "))
        self.assertIn("Person0", red2)
        self.assertNotIn("Grace Wanjiru", red2)

    def test_sender_after_line_without_colon(self):
        """A sender label on a later line is found even when an earlier line has no colon."""
        namemap = {"map": {}, "count": 0}
        redacted = redact_text("Hello team\n[1/1/24, 9:00] Grace: we held the outreach", namemap)
        self.assertEqual(redacted, "Hello team\n[1/1/24, 9:00] Person0: we held the outreach")

    def test_heading_without_timestamp_is_not_a_sender(self):
        """'Update:' at the start of a line is a heading, not a person, unless bare sender labels are enabled."""
        text = "Update: 40 women came.\nNext Update will follow."
        self.assertEqual(redact_text(text, {"map": {}, "count": 0}), text)

    def test_bare_senders_for_dialogue_datasets(self):
        redacted = redact_text("Amanda: hi\nTom: hey Amanda", {"map": {}, "count": 0}, bare_senders=True)
        self.assertEqual(redacted, "Person0: hi\nPerson1: hey Person0")

    def test_excluded_headers_preserved(self):
        namemap = {"map": {}, "count": 0}
        text = "SUBREDDIT: KenyaHealth\nTITLE: Outreach Report\nPOST: Everything went well."
        redacted = redact_text(text, namemap, bare_senders=True)
        self.assertIn("SUBREDDIT:", redacted)
        self.assertIn("TITLE:", redacted)
        self.assertIn("POST:", redacted)
        self.assertNotIn("Person0", redacted)

    def test_scan_batch(self):
        batch = {
            "text": [
                "Alice: My email is alice@test.com",
                "Bob: Call me at +254712345678",
            ]
        }
        res = scan_batch(batch)
        self.assertIn("text", res)
        self.assertEqual(len(res["text"]), 2)
        self.assertNotIn("alice@test.com", res["text"][0])
        self.assertNotIn("+254712345678", res["text"][1])

    def test_names_learned_anywhere_in_a_long_dataset(self):
        """A name introduced in the last of 1,200 messages is also hidden in the first."""
        texts = ["Thanks Rose for the tea."] + ["ok"] * 1198 + ["Hi all, Rose Njeri here."]
        processed = process_dataset_pii(Dataset.from_dict({"text": texts}))
        self.assertNotIn("Rose", processed[0]["text"])
        self.assertNotIn("Rose", processed[-1]["text"])

    def test_process_dataset_pii(self):
        ds = Dataset.from_list([
            {"text": "[01/02/24, 09:00] David: We delivered supplies to the clinic."},
            {"text": "[01/02/24, 09:05] Sarah: David called me earlier."},
        ])
        processed = process_dataset_pii(ds)
        self.assertEqual(len(processed), 2)
        self.assertTrue(processed[0]["text"].startswith("[01/02/24, 09:00] Person0: "))
        self.assertTrue(processed[1]["text"].startswith("[01/02/24, 09:05] Person1: "))
        self.assertIn("Person0", processed[1]["text"])


if __name__ == "__main__":
    unittest.main()
