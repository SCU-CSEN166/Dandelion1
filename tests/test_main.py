"""Unit tests for main.py's argument checks and --table_out columns (no models are loaded)."""
import contextlib
import io
import os
import sys
import unittest

from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from main import TABLE_OUT_COLUMNS, parse_args, table_out_view


class TestMain(unittest.TestCase):

    def test_summarizer_and_output_go_together(self):
        for argv in (["-d", "x.csv", "-s", "facebook/bart-large-cnn"], ["-d", "x.csv", "-o", "summary.txt"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(argv)
        args = parse_args(["-d", "x.csv", "-s", "facebook/bart-large-cnn", "-o", "summary.txt"])
        self.assertEqual(args.output, "summary.txt")
        self.assertIsNone(parse_args(["-d", "x.csv"]).summarizer)

    def test_outputs_that_need_relevance_turn_on_the_classifier(self):
        self.assertFalse(parse_args(["-d", "x.csv", "--table_out", "t.csv"]).classify_relevance)
        for flag in (["--ta_table_out", "ta.csv"], ["--review_out", "r.csv"], ["--followups_out", "f.csv"],
                     ["--protect_sensitive"], ["--filter_relevant"], ["--classify_relevance"]):
            self.assertTrue(parse_args(["-d", "x.csv"] + flag).classify_relevance, flag)

    def test_table_out_keeps_only_pipeline_columns(self):
        ds = Dataset.from_list([{
            "message_id": 1, "sender": "Grace Wanjiru", "phone": "0712 345 678", "original_text": "Grace here",
            "text": "Person0 here", "pii_items": "Grace", "pii_count": 1, "pii_notes": "", "relevance": "Event",
        }])
        view = table_out_view(ds)
        self.assertEqual(view.column_names, ["message_id", "text", "pii_count", "pii_notes", "relevance"])
        self.assertTrue(set(view.column_names) <= set(TABLE_OUT_COLUMNS))


if __name__ == "__main__":
    unittest.main()
