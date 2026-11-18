"""Unit tests for the TA-format table builder and the context-based redaction it relies on.

All messages here are made up for the tests (not taken from the hidden test set).
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from redact import redact_line, scan_batch
from ta_table import TA_COLUMNS, build_ta_table, find_participants, to_ta_tags, write_table


def _table(messages):
    batch = scan_batch({"text": messages}, {"map": {}, "count": 0})
    rows = [{"message_id": i + 1, "text": t, "original_text": m, "pii_items": items, "pii_count": n}
            for i, (m, t, items, n) in enumerate(zip(messages, batch["text"], batch["pii_items"], batch["pii_count"]))]
    return build_ta_table(rows)


class TestContextRedaction(unittest.TestCase):

    def test_titles_introductions_and_later_mentions(self):
        namemap = {"map": {}, "count": 0}
        text, items, _ = redact_line(
            "Good morning, this is Amina Hassan. Nurse Peter Odhiambo helped. Amina will send the list.", namemap)
        self.assertNotIn("Amina", text)
        self.assertNotIn("Odhiambo", text)
        self.assertIn("Nurse Person", text)
        self.assertEqual(len(items), 2)  # "Amina" is the same person as "Amina Hassan"

    def test_name_here_and_group_keep_following_words(self):
        namemap = {"map": {}, "count": 0}
        text, _, _ = redact_line("Hi all, Joy Akinyi here, from the Upendo women's group meeting.", namemap)
        self.assertIn(" here,", text)
        self.assertIn("women's group meeting", text)
        self.assertNotIn("Joy Akinyi", text)
        self.assertNotIn("Upendo", text)

    def test_facilities(self):
        namemap = {"map": {}, "count": 0}
        text, items, _ = redact_line("We met at Kibera Health Centre and later at Mathare dispensary.", namemap)
        self.assertNotIn("Kibera", text)
        self.assertNotIn("Mathare", text)
        self.assertIn("dispensary", text)  # generic noun kept, name hidden
        self.assertIn("Kibera Health Centre", items)

    def test_no_redaction_of_ordinary_words(self):
        namemap = {"map": {}, "count": 0}
        text = "Just adding to her report: it's Saturday and everyone here is fine. Thank you."
        self.assertEqual(redact_line(text, namemap)[0], text)


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

    def test_written_table_has_no_unredacted_text(self):
        messages = ["Hi, this is Grace Wanjiru. Call me on 0712 345 678. We held a clinic and 40 women came."]
        table = _table(messages)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ta.csv")
            write_table(table, path)
            with open(path, encoding="utf-8") as f:
                written = f.read()
        self.assertNotIn("transcript", written.splitlines()[0])
        self.assertNotIn("pii_items", written.splitlines()[0])
        for raw in ("Grace", "Wanjiru", "0712 345 678"):
            self.assertNotIn(raw, written)

    def test_tags(self):
        self.assertEqual(to_ta_tags("Person0 met Person1 at Place0, call [PHONE]"),
                         "[NAME] met [NAME] at [LOCATION], call [PHONE]")
        # an email and a phone number of the same length keep their own tags
        text = redact_line("mail abcdef@gh.io or call 0722 111 222", {"map": {}, "count": 0})[0]
        self.assertEqual(to_ta_tags(text), "mail [EMAIL] or call [PHONE]")

    def test_classifier_label_decides_relevance(self):
        """With a classifier label the rules are not consulted; without one they are the fallback."""
        report = "We held a nutrition session and 30 women came."
        rows = [{"message_id": 1, "text": report, "original_text": report, "relevance": "Chit-chat"},
                {"message_id": 2, "text": "Thanks all!", "original_text": "Thanks all!", "relevance": "Event"},
                {"message_id": 3, "text": report, "original_text": report}]
        table = build_ta_table(rows)
        self.assertEqual(list(table["relevance"]), ["Chit-chat", "Event", "Event"])
        self.assertEqual(table.loc[1, "activity_type"], "Other field activity")

    def test_participants_left_blank_when_qa_and_rule_disagree(self):
        text = "We held a nutrition session and 30 women came."
        rows = [{"message_id": 1, "text": text, "original_text": text, "participants": "45", "pii_count": 0},
                {"message_id": 2, "text": text, "original_text": text, "participants": "30", "pii_count": 0}]
        table = build_ta_table(rows)
        self.assertEqual(table.loc[0, "participants"], "")  # QA 45 vs rule 30
        self.assertEqual(table.loc[1, "participants"], 30)


if __name__ == "__main__":
    unittest.main()
