"""Unit tests for speech-error handling in redaction, the review queue, follow-ups and sensitive-topic protection.

All messages here are made up for the tests (not taken from the hidden test set).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from insights import (build_followups, build_review_table, confidence_from, leftover_identifiers, protect_text,
                      sensitive_topic, split_sentences)
from redact import _edit_distance, scan_batch
from ta_table import build_ta_table


def _tables(messages):
    batch = scan_batch({"text": messages}, {"map": {}, "count": 0})
    rows = [{"message_id": i + 1, "text": t, "original_text": m, "pii_items": items, "pii_count": n, "pii_notes": notes}
            for i, (m, t, items, n, notes) in enumerate(
                zip(messages, batch["text"], batch["pii_items"], batch["pii_count"], batch["pii_notes"]))]
    ta = build_ta_table(rows)
    return ta, build_review_table(ta, rows), build_followups(ta)


class TestSpeechErrorRedaction(unittest.TestCase):

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


class TestReviewQueue(unittest.TestCase):

    def test_confident_event_needs_no_review(self):
        _, review, _ = _tables(["Hello team, we held the kitchen garden training today and twenty farmers came."])
        self.assertEqual(review.loc[0, "confidence_level"], "High")
        self.assertFalse(review.loc[0, "needs_review"])

    def test_doubts_lower_confidence(self):
        _, review, _ = _tables([
            "We held the nutrition session today, it went well but I did not count people.",
            "Hi ladies, we talked about nutrition over tea. Greetings from Wekesa.",
        ])
        self.assertIn("no attendance count", review.loc[0, "review_reasons"])
        self.assertLess(review.loc[0, "confidence"], 1.0)
        self.assertIn("possible missed event", review.loc[1, "review_reasons"])
        self.assertIn("place guessed from context: Place0", review.loc[1, "review_reasons"])
        self.assertNotIn("Wekesa", review.loc[1, "review_reasons"])  # reasons name the alias, never the original
        self.assertTrue(review.loc[1, "needs_review"])

    def test_confidence_counts_each_doubt_once(self):
        self.assertEqual(confidence_from([]), 1.0)
        self.assertEqual(confidence_from(["place guessed from context: A", "place guessed from context: B"]), 0.9)

    def test_leftover_identifiers(self):
        self.assertEqual(leftover_identifiers("Thanks [NAME]. We met Otieno on Monday at [LOCATION]."), ["Otieno"])
        self.assertIn("a long number", leftover_identifiers("call 0722 111 222 today"))


class TestSensitiveAndFollowups(unittest.TestCase):

    MESSAGES = [
        "Hi team, this is Amina Hassan. We did the HIV testing day at Kibera Health Centre and 30 people came. "
        "Two were referred to the county hospital. One of them is Mr. John Otieno and he has agreed to go on Friday. "
        "His number is 0722 111 222. I will send the report tomorrow.",
        "Morning all, we held the kitchen garden training and 12 farmers came. Next month we will visit their homes.",
    ]

    def test_sentence_split_keeps_titles(self):
        self.assertEqual(len(split_sentences("One of them is Mrs. [NAME], and she agreed. Her number is [PHONE].")), 2)

    def test_sensitive_topic(self):
        self.assertEqual(sensitive_topic("community dialogue on gender-based violence"), "Gender-based violence")
        self.assertEqual(sensitive_topic("kitchen garden training"), "")

    def test_individual_sentences_withheld(self):
        _, review, _ = _tables(self.MESSAGES)
        report = review.loc[0, "report_text"]
        self.assertEqual(review.loc[0, "sensitive_topic"], "HIV")
        self.assertTrue(review.loc[0, "needs_review"])
        self.assertIn("30 people came", report)
        self.assertIn("Two were referred", report)
        self.assertNotIn("One of them", report)
        self.assertNotIn("[PHONE]", report)
        self.assertIn("withheld", report)
        self.assertEqual(review.loc[1, "sensitive_topic"], "")
        self.assertEqual(protect_text("Nothing sensitive here.", ""), "Nothing sensitive here.")

    def test_followups(self):
        _, _, followups = _tables(self.MESSAGES)
        by_category = dict(zip(followups.category, followups.access))
        self.assertEqual(by_category["Referral"], "general")
        self.assertEqual(by_category["Report to send"], "general")
        self.assertEqual(by_category["Next visit or meeting"], "general")
        restricted = followups[followups.access.str.startswith("restricted")]
        self.assertEqual(len(restricted), 1)
        self.assertIn("agreed to go", restricted.iloc[0].follow_up)
        self.assertNotIn("Otieno", " ".join(followups.follow_up))


if __name__ == "__main__":
    unittest.main()
