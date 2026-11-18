"""Unit tests for the PII redaction stage."""
import sys
import os
import unittest
from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from redact import redact_line_with_items, scan_batch, process_dataset_pii


def redact_line(text, namemap):
    return redact_line_with_items(text, namemap)[0]


class TestRedact(unittest.TestCase):

    def test_redact_email(self):
        namemap = {"map": {}, "count": 0}
        text = "Contact me at user.name@example.org or info@dandelionafrica.org please."
        redacted = redact_line(text, namemap)
        self.assertNotIn("user.name@example.org", redacted)
        self.assertNotIn("info@dandelionafrica.org", redacted)
        self.assertIn("*" * len("user.name@example.org"), redacted)
        self.assertIn("*" * len("info@dandelionafrica.org"), redacted)

    def test_redact_phone_international(self):
        namemap = {"map": {}, "count": 0}
        text = "Call +1 (555) 234-5678 or 555-876-5432 today."
        redacted = redact_line(text, namemap)
        self.assertNotIn("555) 234-5678", redacted)
        self.assertNotIn("555-876-5432", redacted)
        self.assertIn("****", redacted)

    def test_redact_phone_kenyan(self):
        namemap = {"map": {}, "count": 0}
        text = "Reach Sister Nancy at +254 712 345 678 or 07560080089."
        redacted = redact_line(text, namemap)
        self.assertNotIn("254 712 345 678", redacted)
        self.assertNotIn("07560080089", redacted)
        self.assertIn("****", redacted)

    def test_sender_anonymization_and_body_mentions(self):
        namemap = {"map": {}, "count": 0}
        line1 = "Grace Wanjiru: We served 40 women at the dispensary today."
        red1 = redact_line(line1, namemap)
        self.assertTrue(red1.startswith("Person0: "))
        self.assertNotIn("Grace Wanjiru:", red1)

        line2 = "John Doe: I spoke with Grace Wanjiru earlier about the medicine."
        red2 = redact_line(line2, namemap)
        self.assertTrue(red2.startswith("Person1: "))
        self.assertIn("Person0", red2)
        self.assertNotIn("Grace Wanjiru", red2)

    def test_excluded_headers_preserved(self):
        namemap = {"map": {}, "count": 0}
        text = "SUBREDDIT: KenyaHealth\nTITLE: Outreach Report\nPOST: Everything went well."
        redacted = redact_line(text, namemap)
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

    def test_process_dataset_pii(self):
        ds = Dataset.from_list([
            {"text": "David: We delivered supplies to the clinic."},
            {"text": "Sarah: David called me earlier."},
        ])
        processed = process_dataset_pii(ds)
        self.assertEqual(len(processed), 2)
        self.assertTrue(processed[0]["text"].startswith("Person0: "))
        self.assertTrue(processed[1]["text"].startswith("Person1: "))
        self.assertIn("Person0", processed[1]["text"])


if __name__ == "__main__":
    unittest.main()
