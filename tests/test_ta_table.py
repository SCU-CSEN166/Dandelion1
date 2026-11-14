"""Unit tests for the TA-format table builder and the context-based redaction it relies on.

All messages here are made up for the tests (not taken from the hidden test set).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from redact import redact_line_with_items, scan_batch
from ta_table import TA_COLUMNS, build_ta_table, find_participants, to_ta_tags


def _table(messages):
    batch = scan_batch({"text": messages}, {"map": {}, "count": 0})
    rows = [{"message_id": i + 1, "text": t, "transcript": m, "pii_items": items, "pii_count": n}
            for i, (m, t, items, n) in enumerate(zip(messages, batch["text"], batch["pii_items"], batch["pii_count"]))]
    return build_ta_table(rows)


class TestContextRedaction(unittest.TestCase):

    def test_titles_introductions_and_later_mentions(self):
        namemap = {"map": {}, "count": 0}
        text, items = redact_line_with_items(
            "Good morning, this is Amina Hassan. Nurse Peter Odhiambo helped. Amina will send the list.", namemap)
        self.assertNotIn("Amina", text)
        self.assertNotIn("Odhiambo", text)
        self.assertIn("Nurse Person", text)
        self.assertEqual(len(items), 2)  # "Amina" is the same person as "Amina Hassan"

    def test_name_here_and_group_keep_following_words(self):
        namemap = {"map": {}, "count": 0}
        text, _ = redact_line_with_items("Hi all, Joy Akinyi here, from the Upendo women's group meeting.", namemap)
        self.assertIn(" here,", text)
        self.assertIn("women's group meeting", text)
        self.assertNotIn("Joy Akinyi", text)
        self.assertNotIn("Upendo", text)

    def test_facilities(self):
        namemap = {"map": {}, "count": 0}
        text, items = redact_line_with_items("We met at Kibera Health Centre and later at Mathare dispensary.", namemap)
        self.assertNotIn("Kibera", text)
        self.assertNotIn("Mathare", text)
        self.assertIn("dispensary", text)  # generic noun kept, name hidden
        self.assertIn("Kibera Health Centre", items)

    def test_no_redaction_of_ordinary_words(self):
        namemap = {"map": {}, "count": 0}
        text = "Just adding to her report: it's Saturday and everyone here is fine. Thank you."
        self.assertEqual(redact_line_with_items(text, namemap)[0], text)


class TestTaTable(unittest.TestCase):

    def test_columns(self):
        self.assertEqual(list(_table(["Hello team"]).columns), TA_COLUMNS)

    def test_participant_patterns(self):
        self.assertEqual(find_participants("In total, thirty-two mothers came today."), 32)
        self.assertEqual(find_participants("We screened 15 women at the camp."), 15)
        self.assertEqual(find_participants("Attendance was forty at the meeting."), 40)
        self.assertIsNone(find_participants("We waited two hours for the bus."))

    def test_event_chitchat_and_duplicates(self):
        df = _table([
            "Hello team, this is Amina Hassan. We held the immunization outreach and fifty children came.",
            "Hi ladies, who is bringing the tent on Saturday? Call me on 0722 111 222.",
            "Good evening, it's Joy Akinyi. Adding to the immunization report: we vaccinated 50 children today.",
            "Morning, we trained 20 farmers on composting near the river.",
        ])
        self.assertEqual(list(df.relevance), ["Event", "Chit-chat", "Event", "Event"])
        self.assertEqual(df.loc[0, "activity_type"], "Immunization outreach")
        self.assertEqual(list(df.participants.astype(str)), ["50", "", "50", "20"])
        self.assertEqual(list(df.duplicate_of), ["none", "none", "1", "none"])
        self.assertIn("[PHONE]", df.loc[1, "expected_anonymized_text"])
        self.assertIn("[NAME]", df.loc[0, "expected_anonymized_text"])

    def test_tags(self):
        self.assertEqual(to_ta_tags("Person0 met Person1 at Place0, call ************", "0722 111 222"),
                         "[NAME] met [NAME] at [LOCATION], call [PHONE]")
        self.assertEqual(to_ta_tags("mail ************", "joy@site.com"), "mail [EMAIL]")


if __name__ == "__main__":
    unittest.main()
