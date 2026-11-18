"""Unit tests for the QA extraction stage."""
import sys
import os
import unittest
from datasets import Dataset, DatasetDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extract import (
    DEFAULT_EVENT_QUESTION,
    DEFAULT_PARTICIPANTS_QUESTION,
    parse_participant_count,
    clean_event_label,
    extract_dataset_info,
)


class DummyQAPipeline:
    """Mock Question-Answering pipeline for fast, offline testing."""

    def __call__(self, inputs, batch_size=None, **kwargs):
        single = isinstance(inputs, dict)
        items = [inputs] if single else inputs
        results = []

        for item in items:
            question = item.get("question", "").lower()
            context = item.get("context", "").lower()

            # Event question
            if "event" in question or "activity" in question:
                if "family planning" in context:
                    results.append({"answer": "today's family planning outreach", "score": 0.92, "start": 10, "end": 42})
                elif "table banking" in context:
                    results.append({"answer": "our monthly table banking day", "score": 0.88, "start": 5, "end": 35})
                elif "cervical cancer" in context:
                    results.append({"answer": "cervical cancer screening camp.", "score": 0.95, "start": 12, "end": 43})
                else:
                    results.append({"answer": "", "score": 0.01, "start": 0, "end": 0})

            # Participants question
            elif "people" in question or "participant" in question or "attend" in question:
                if "42 women" in context:
                    results.append({"answer": "42 women", "score": 0.94, "start": 20, "end": 28})
                elif "25 members" in context:
                    results.append({"answer": "25 members", "score": 0.91, "start": 15, "end": 25})
                elif "sixty girls" in context:
                    results.append({"answer": "sixty girls", "score": 0.89, "start": 10, "end": 21})
                else:
                    results.append({"answer": "", "score": 0.01, "start": 0, "end": 0})
            else:
                results.append({"answer": "", "score": 0.0, "start": 0, "end": 0})

        return results[0] if single else results


class TestExtraction(unittest.TestCase):

    def setUp(self):
        self.mock_qa = DummyQAPipeline()
        self.sample_rows = [
            {
                "text": "We held a family planning outreach today. In total, 42 women came.",
                "relevance": "relevant field activity",
            },
            {
                "text": "Today was our monthly table banking day. 25 members attended.",
                "relevance": "relevant field activity",
            },
            {
                "text": "E, congratulations on the new baby!! Bring your own chairs.",
                "relevance": "casual chit-chat",
            },
            {
                "text": "",
                "relevance": "casual chit-chat",
            },
        ]
        self.dataset = Dataset.from_list(self.sample_rows)
        self.dataset_dict = DatasetDict({
            "train": Dataset.from_list(self.sample_rows),
            "test": Dataset.from_list(self.sample_rows[:2]),
        })

    def test_extract_dataset_info_adds_columns(self):
        """Test that extract_dataset_info adds event_type and participants columns."""
        result = extract_dataset_info(
            self.dataset,
            qa_pipeline=self.mock_qa,
            event_type_col="event_type",
            participants_col="participants",
        )

        self.assertIn("event_type", result.column_names)
        self.assertIn("participants", result.column_names)
        self.assertEqual(len(result), len(self.dataset))

        # Row 0: Family planning outreach, 42
        self.assertEqual(result[0]["event_type"], "Family planning outreach")
        self.assertEqual(result[0]["participants"], "42")

        # Row 1: Table banking day, 25
        self.assertEqual(result[1]["event_type"], "Table banking day")
        self.assertEqual(result[1]["participants"], "25")

        # Row 2 (Chit-chat): should be skipped because of relevance column
        self.assertEqual(result[2]["event_type"], "")
        self.assertEqual(result[2]["participants"], "")

        # Row 3 (Empty line): should be empty
        self.assertEqual(result[3]["event_type"], "")
        self.assertEqual(result[3]["participants"], "")

    def test_extract_dataset_dict(self):
        """Test extraction across all splits of a DatasetDict."""
        result = extract_dataset_info(
            self.dataset_dict,
            qa_pipeline=self.mock_qa,
        )
        self.assertIsInstance(result, DatasetDict)
        self.assertIn("event_type", result["train"].column_names)
        self.assertIn("participants", result["train"].column_names)
        self.assertIn("event_type", result["test"].column_names)
        self.assertIn("participants", result["test"].column_names)

        self.assertEqual(result["train"][0]["event_type"], "Family planning outreach")
        self.assertEqual(result["test"][0]["event_type"], "Family planning outreach")

    def test_custom_column_names(self):
        """Test custom column names for event type and participants."""
        result = extract_dataset_info(
            self.dataset,
            qa_pipeline=self.mock_qa,
            event_type_col="activity_type",
            participants_col="attendee_count",
        )
        self.assertIn("activity_type", result.column_names)
        self.assertIn("attendee_count", result.column_names)
        self.assertEqual(result[0]["activity_type"], "Family planning outreach")
        self.assertEqual(result[0]["attendee_count"], "42")

    def test_ignore_relevance_flag(self):
        """Test that ignore_relevance=True queries lines even if marked chit-chat."""
        # Row with chit-chat relevance tag but contains event text
        special_ds = Dataset.from_list([
            {
                "text": "We did cervical cancer screening camp today.",
                "relevance": "casual chit-chat",
            }
        ])
        # By default (use_relevance=True), it skips chit-chat
        res_skipped = extract_dataset_info(special_ds, qa_pipeline=self.mock_qa, use_relevance=True)
        self.assertEqual(res_skipped[0]["event_type"], "")

        # With use_relevance=False, it extracts despite the tag
        res_forced = extract_dataset_info(special_ds, qa_pipeline=self.mock_qa, use_relevance=False)
        self.assertEqual(res_forced[0]["event_type"], "Cervical cancer screening camp")

    def test_parse_participant_count(self):
        """Test parsing various participant formats to normalized integers."""
        self.assertEqual(parse_participant_count("42 women"), "42")
        self.assertEqual(parse_participant_count("served 25 members"), "25")
        self.assertEqual(parse_participant_count("1,250 people"), "1250")
        self.assertEqual(parse_participant_count("forty-two"), "42")
        self.assertEqual(parse_participant_count("twenty five"), "25")
        self.assertEqual(parse_participant_count("sixty"), "60")
        self.assertEqual(parse_participant_count(""), "")
        self.assertEqual(parse_participant_count(None), "")

    def test_clean_event_label(self):
        """Test cleaning event labels."""
        self.assertEqual(clean_event_label("today's family planning outreach."), "Family planning outreach")
        self.assertEqual(clean_event_label("the cervical cancer screening camp;"), "Cervical cancer screening camp")
        self.assertEqual(clean_event_label("our monthly table banking day"), "Table banking day")
        self.assertEqual(clean_event_label(""), "")



if __name__ == "__main__":
    unittest.main()
