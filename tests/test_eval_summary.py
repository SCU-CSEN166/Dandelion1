"""Unit tests for eval_summary.py's parsing and scoring (no judge model is loaded)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval")))

from eval_summary import key_facts_from_key, parse_numbers, parse_yes, report_sentences, scores


class TestEvalSummary(unittest.TestCase):

    def test_report_sentences_drop_headings_and_markdown(self):
        report = ("**Field Outcomes Report**\n\n1. **Practical Actions Completed:**\n"
                  "- Person0 held a family planning outreach. 42 women attended.\n"
                  "2. Blockers & Technical Issues: None.\n### Pending Follow-ups:\n* Mrs. Person1 will send the register.")
        self.assertEqual(report_sentences(report), ["Person0 held a family planning outreach.",
                                                    "42 women attended.", "Mrs. Person1 will send the register."])

    def test_parse_judge_answers(self):
        self.assertTrue(parse_yes("Yes."))
        self.assertFalse(parse_yes("No, the number is wrong"))
        self.assertFalse(parse_yes("Not yes"))
        self.assertEqual(parse_numbers("2, 5 and 9", 5), [2, 5])
        self.assertEqual(parse_numbers("None", 5), [])

    def test_scores(self):
        s = scores([True, True, False, True], [[1], [], [1, 2]], 4)
        self.assertEqual(s, {"faithfulness": 0.75, "completeness": 0.667, "conciseness": 0.5})
        self.assertEqual(scores([], [], 0), {"faithfulness": None, "completeness": None, "conciseness": None})

    def test_key_facts_skip_duplicates_and_chitchat(self):
        facts = key_facts_from_key(os.path.join(os.path.dirname(__file__), "..", "data", "hidden_test_outputs.xlsx"))
        self.assertEqual(len(facts), 6)  # 8 Events, 2 of them duplicates
        self.assertIn("Family planning outreach with 42 participants", facts)


if __name__ == "__main__":
    unittest.main()
