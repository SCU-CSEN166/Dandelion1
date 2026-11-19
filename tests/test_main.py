"""Unit tests for main.py's arguments and the facts it gives the summarizer (no models are loaded)."""
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from main import parse_args, report_facts


class TestMain(unittest.TestCase):

    def test_dataset_and_out_are_required(self):
        for argv in (["-d", "x.csv"], ["-o", "outputs"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(argv)
        args = parse_args(["-d", "x.csv", "-o", "outputs"])
        self.assertEqual(args.out, "outputs")
        self.assertIsNone(args.summarizer)

    def test_report_facts(self):
        rows = [{"event_type": "Kitchen garden training", "participants": "20"},
                {"event_type": "", "participants": ""},
                {"event_type": "", "participants": "12"}]
        self.assertEqual(report_facts(rows), ["- Event: Kitchen garden training | Attendees/Participants: 20",
                                              "- Event: Field activity | Attendees/Participants: 12"])


if __name__ == "__main__":
    unittest.main()
