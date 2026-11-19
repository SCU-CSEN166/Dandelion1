"""Unit tests for the QA extraction stage."""
import sys
import os
import unittest
from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extract import (
    DEFAULT_EVENT_QUESTION,
    DEFAULT_PARTICIPANTS_QUESTION,
    parse_participant_count,
    clean_event_label,
    extract_dataset_info,
    find_participants,
    first_number,
    activity_type,
    classify_activities,
    ACTIVITY_CATEGORIES,
    ACTIVITY_MIN_SCORE,
    OTHER_ACTIVITY,
)


class DummyCategoryPipeline:
    """Mock zero-shot pipeline: picks 'Tree planting' for messages about trees, else the first category."""

    def __init__(self):
        self.calls = 0

    def __call__(self, inputs, candidate_labels=None, hypothesis_template=None, multi_label=False, batch_size=None):
        self.calls += 1
        out = []
        for text in [inputs] if isinstance(inputs, str) else inputs:
            top = "Tree planting" if "tree" in text.lower() else candidate_labels[0]
            out.append({"labels": [top] + [c for c in candidate_labels if c != top], "scores": [0.8] + [0.01] * 22})
        return out[0] if isinstance(inputs, str) else out


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

    def test_hundred_and(self):
        """'one hundred and nine' is 109; 'and' joins numbers only after hundred / thousand."""
        self.assertEqual(first_number("one hundred and nine people came"), 109)
        self.assertEqual(find_participants("One hundred and thirty-four people came today"), 134)
        self.assertEqual(find_participants("between forty and fifty people came"), 50)

    def test_classify_activities(self):
        ds = Dataset.from_list([
            {"text": "We planted trees by the school today.", "relevance": "Event"},
            {"text": "Good morning all!", "relevance": "Chit-chat"},
        ])
        result = classify_activities(ds, classifier=DummyCategoryPipeline())
        self.assertEqual(result["activity_category"], ["Tree planting", ""])  # Chit-chat gets no category
        self.assertEqual(result["activity_score"], [0.8, 0.0])

    def test_activity_type_combination(self):
        text = "We held the immunization day."
        # a confident category wins; below ACTIVITY_MIN_SCORE the topic rule, then QA, then "Other"
        self.assertEqual(activity_type(text, "", "Immunization outreach", ACTIVITY_MIN_SCORE), "Immunization outreach")
        self.assertEqual(activity_type(text, "", "Tree planting", ACTIVITY_MIN_SCORE - 0.01), "Immunization outreach")
        self.assertEqual(activity_type("We met.", "borehole meeting", "Tree planting", 0.05), "Borehole meeting")
        self.assertEqual(activity_type("We met.", "", "", 0.0), OTHER_ACTIVITY)
        self.assertIn("Immunization outreach", ACTIVITY_CATEGORIES)

    def test_participant_patterns(self):
        self.assertEqual(find_participants("In total, thirty-two mothers came today."), 32)
        self.assertEqual(find_participants("We screened 15 women at the camp."), 15)
        self.assertEqual(find_participants("Attendance was forty at the meeting."), 40)
        self.assertIsNone(find_participants("We waited two hours for the bus."))

    def test_parse_participant_count(self):
        """Test parsing various participant formats to normalized integers."""
        self.assertEqual(parse_participant_count("42 women"), "42")
        self.assertEqual(parse_participant_count("served 25 members"), "25")
        self.assertEqual(parse_participant_count("1,250 people"), "1250")
        self.assertEqual(parse_participant_count("forty-two"), "42")
        self.assertEqual(parse_participant_count("twenty five"), "25")
        self.assertEqual(parse_participant_count("sixty"), "60")
        self.assertEqual(parse_participant_count("two thousand five hundred people"), "2500")
        self.assertEqual(parse_participant_count("the 42 women and one nurse"), "42")  # digits win over number words
        # numbers that are not head counts
        self.assertEqual(parse_participant_count("one of the mothers"), "")
        self.assertEqual(parse_participant_count("the 2024 outreach"), "")
        self.assertEqual(parse_participant_count("at 10:30"), "")
        self.assertEqual(parse_participant_count("9am"), "")
        self.assertEqual(parse_participant_count("500 shillings"), "")
        self.assertEqual(parse_participant_count("a few elders"), "a few elders")
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
