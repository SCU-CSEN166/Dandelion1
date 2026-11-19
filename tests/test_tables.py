"""Unit tests for the output tables (tables.py) and the context-based redaction they rely on.

All messages here are made up for the tests (not taken from the hidden test set). Relevance labels are given, as the
classifier would give them in the pipeline.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from redact import _edit_distance, redact_line, scan_batch
from tables import (FOLLOWUP_COLUMNS, MESSAGE_COLUMNS, build_followups, build_message_table, confidence_from,
                    leftover_identifiers, protect_text, sensitive_topic, split_sentences, write_table)


def _tables(messages, relevance, qa_participants=None):
    batch = scan_batch({"text": messages}, {"map": {}, "count": 0})
    rows = [{"message_id": i + 1, "text": t, "original_text": m, "pii_count": n, "pii_notes": notes,
             "relevance": r, "relevance_score": 0.999 if r == "Event" else 0.1}
            for i, (m, t, n, notes, r) in enumerate(
                zip(messages, batch["text"], batch["pii_count"], batch["pii_notes"], relevance))]
    for row, qa in zip(rows, qa_participants or []):
        row["participants"] = qa
    table = build_message_table(rows)
    return table, build_followups(table)


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

    def test_places_and_people_from_context(self):
        out = scan_batch({"text": ["The bus from Zolani broke down near Mtwapa. Send the money to Halima on M-Pesa."]})
        text = out["text"][0]
        for hidden in ("Zolani", "Mtwapa", "Halima"):
            self.assertNotIn(hidden, text)
        self.assertIn("M-Pesa", text)
        self.assertIn("place guessed from context", out["pii_notes"][0])

    def test_misspelled_known_place(self):
        out = scan_batch({"text": ["We met the group near Girissa town.", "Then we drove back to Garissa."]})
        self.assertNotIn("Girissa", out["text"][0])
        self.assertIn("possible misspelling of a known place: Place", out["pii_notes"][0])
        self.assertNotIn("Girissa", out["pii_notes"][0])  # notes name the alias, not the original
        first = out["text"][0].split("near ")[1].split()[0]
        self.assertIn(first, out["text"][1])  # the misspelling and the real name share one alias

    def test_name_learned_later_is_hidden_earlier(self):
        out = scan_batch({"text": ["Wanjala will bring the chairs.", "Hello, this is Peter Wanjala from the office."]})
        self.assertNotIn("Wanjala", out["text"][0])

    def test_ordinary_text_untouched(self):
        text = "I was at home in January. Please bring sugar to the meeting at church on Sunday."
        self.assertEqual(scan_batch({"text": [text]})["text"][0], text)

    def test_form_labels_are_not_names(self):
        text = "Contact Number: 0722 111 222\nDate of Birth: unknown"
        out = scan_batch({"text": [text]})["text"][0]
        self.assertIn("Contact Number:", out)
        self.assertIn("Date of Birth:", out)

    def test_edit_distance(self):
        self.assertEqual(_edit_distance("girissa", "garissa", 2), 1)
        self.assertEqual(_edit_distance("kampiamoto", "kampiyamoto", 2), 1)
        self.assertGreater(_edit_distance("nairobi", "mombasa", 2), 2)


class TestMessageTable(unittest.TestCase):

    def test_events_chitchat_and_duplicates(self):
        table, _ = _tables([
            "Hello team, this is Amina Hassan. We held the immunization outreach and fifty children came.",
            "Hi ladies, who is bringing the tent on Saturday? Call me on 0722 111 222.",
            "Good evening, it's Joy Akinyi. Adding to the immunization report: we vaccinated 50 children today.",
            "Morning, we trained 20 farmers on composting near the river.",
        ], ["Event", "Chit-chat", "Event", "Event"])
        self.assertEqual(table.loc[0, "activity_type"], "Immunization outreach")
        self.assertEqual(list(table.participants.astype(str)), ["50", "", "50", "20"])
        self.assertEqual(list(table.duplicate_of), ["none", "none", "1", "none"])
        self.assertIn("[PHONE]", table.loc[1, "text"])
        self.assertIn("Person0", table.loc[0, "text"])  # one alias style everywhere: Person0 / Place0

    def test_classifier_category_and_disagreement(self):
        """The classifier's category fills activity_type; a topic rule that says otherwise is a review reason."""
        text = "We held the immunization day and 40 children came."
        rows = [{"message_id": 1, "text": text, "original_text": text, "relevance": "Event",
                 "activity_category": "Nutrition session", "activity_score": 0.9},
                {"message_id": 2, "text": text, "original_text": text, "relevance": "Event",
                 "activity_category": "Immunization outreach", "activity_score": 0.9}]
        table = build_message_table(rows)
        self.assertEqual(list(table["activity_type"]), ["Nutrition session", "Immunization outreach"])
        self.assertIn("activity types disagree (table Nutrition session, topic rule Immunization outreach)",
                      table.loc[0, "review_reasons"])
        self.assertNotIn("activity types disagree", table.loc[1, "review_reasons"])

    def test_count_from_one_method_only(self):
        text = "We held the immunization day and 40 children came."
        rows = [{"message_id": 1, "text": text, "original_text": text, "relevance": "Event",
                 "event_type": "", "participants": ""}]  # the QA model ran but found no count; the rule did
        table = build_message_table(rows)
        self.assertEqual(table.loc[0, "participants"], 40)
        self.assertIn("head count from one method only (number rule)", table.loc[0, "review_reasons"])

    def test_unrecognised_activity(self):
        table, _ = _tables(["Thanks all!"], ["Event"])
        self.assertEqual(table.loc[0, "activity_type"], "Other field activity")
        self.assertIn("activity type not recognised", table.loc[0, "review_reasons"])

    def test_participants_left_blank_when_qa_and_rule_disagree(self):
        text = "We held a nutrition session and 30 women came."
        table, _ = _tables([text, text], ["Event", "Event"], qa_participants=["45", "30"])
        self.assertEqual(table.loc[0, "participants"], "")  # QA 45 vs rule 30
        self.assertIn("participant counts disagree", table.loc[0, "review_reasons"])
        self.assertEqual(table.loc[1, "participants"], 30)

    def test_written_tables_have_only_listed_columns_and_no_raw_text(self):
        messages = ["Hi, this is Grace Wanjiru. Call me on 0712 345 678. We held a clinic and 40 women came. "
                    "I will send the report tomorrow."]
        table, followups = _tables(messages, ["Event"])
        with tempfile.TemporaryDirectory() as tmp:
            for name, df, columns in (("m.csv", table, MESSAGE_COLUMNS), ("f.csv", followups, FOLLOWUP_COLUMNS)):
                path = os.path.join(tmp, name)
                write_table(df, path, columns)
                with open(path, encoding="utf-8") as f:
                    written = f.read()
                self.assertEqual(written.splitlines()[0].split(","), columns)
                for raw in ("Grace", "Wanjiru", "0712 345 678"):
                    self.assertNotIn(raw, written)


class TestReviewQueue(unittest.TestCase):

    def test_confident_event_needs_no_review(self):
        table, _ = _tables(["Hello team, we held the kitchen garden training today and twenty farmers came."], ["Event"])
        self.assertEqual(table.loc[0, "confidence_level"], "High")
        self.assertFalse(table.loc[0, "needs_review"])

    def test_doubts_lower_confidence(self):
        table, _ = _tables([
            "We held the nutrition session today, it went well but I did not count people.",
            "Hi ladies, we talked about nutrition over tea. Greetings from Wekesa.",
        ], ["Event", "Chit-chat"])
        self.assertIn("no attendance count", table.loc[0, "review_reasons"])
        self.assertLess(table.loc[0, "confidence"], 1.0)
        self.assertIn("possible missed event", table.loc[1, "review_reasons"])
        self.assertIn("place guessed from context: Place0", table.loc[1, "review_reasons"])
        self.assertNotIn("Wekesa", table.loc[1, "review_reasons"])  # reasons name the alias, never the original
        self.assertTrue(table.loc[1, "needs_review"])

    def test_confidence_counts_each_doubt_once(self):
        self.assertEqual(confidence_from([]), 1.0)
        self.assertEqual(confidence_from(["place guessed from context: A", "place guessed from context: B"]), 0.9)

    def test_leftover_identifiers(self):
        self.assertEqual(leftover_identifiers("Thanks Person0. We met Otieno on Monday at Place1."), ["Otieno"])
        self.assertIn("a long number", leftover_identifiers("call 0722 111 222 today"))


class TestSensitiveAndFollowups(unittest.TestCase):

    MESSAGES = [
        "Hi team, this is Amina Hassan. We did the HIV testing day at Kibera Health Centre and 30 people came. "
        "Two were referred to the county hospital. One of them is Mr. John Otieno and he has agreed to go on Friday. "
        "His number is 0722 111 222. I will send the report tomorrow.",
        "Morning all, we held the kitchen garden training and 12 farmers came. Next month we will visit their homes.",
    ]

    def test_sentence_split_keeps_titles(self):
        self.assertEqual(len(split_sentences("One of them is Mrs. Person0, and she agreed. Her number is [PHONE].")), 2)

    def test_sensitive_topic(self):
        self.assertEqual(sensitive_topic("community dialogue on gender-based violence"), "Gender-based violence")
        self.assertEqual(sensitive_topic("kitchen garden training"), "")

    def test_individual_sentences_withheld(self):
        table, _ = _tables(self.MESSAGES, ["Event", "Event"])
        report = table.loc[0, "report_text"]
        self.assertEqual(table.loc[0, "sensitive_topic"], "HIV")
        self.assertTrue(table.loc[0, "needs_review"])
        self.assertIn("30 people came", report)
        self.assertIn("Two were referred", report)
        self.assertNotIn("One of them", report)
        self.assertNotIn("[PHONE]", report)
        self.assertIn("withheld", report)
        self.assertEqual(table.loc[1, "sensitive_topic"], "")
        self.assertEqual(protect_text("Nothing sensitive here.", ""), "Nothing sensitive here.")

    def test_followups(self):
        _, followups = _tables(self.MESSAGES, ["Event", "Event"])
        by_category = dict(zip(followups.category, followups.access))
        self.assertEqual(by_category["Referral"], "general")
        self.assertEqual(by_category["Report to send"], "general")
        self.assertEqual(by_category["Next visit or meeting"], "general")
        restricted = followups[followups.access.str.startswith("restricted")]
        self.assertEqual(len(restricted), 1)
        self.assertIn("agreed to go", restricted.iloc[0].follow_up)
        self.assertNotIn("Otieno", " ".join(followups.follow_up))

    def test_no_followups_from_chitchat(self):
        _, followups = _tables(["Please bring more chairs next week."], ["Chit-chat"])
        self.assertEqual(len(followups), 0)


if __name__ == "__main__":
    unittest.main()
