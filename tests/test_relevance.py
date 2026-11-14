"""Unit tests for the relevance classification stage."""
import sys
import os
import unittest
from unittest.mock import MagicMock
from datasets import Dataset, DatasetDict
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from relevance import (
    DEFAULT_LABELS,
    DEFAULT_HYPOTHESIS_TEMPLATE,
    PRESET_LABELS,
    classify_dataset_relevance,
    process_dataset_relevance,
    classify_line,
    filter_relevant_lines,
    _resolve_text_column,
)


class DummyPipeline:
    """Mock zero-shot classification pipeline for fast, offline testing."""

    def __call__(self, inputs, candidate_labels=None, hypothesis_template=None, multi_label=False, batch_size=None):
        single = isinstance(inputs, str)
        items = [inputs] if single else inputs
        results = []
        for text in items:
            text_lower = text.lower()
            # Simple rule-based mock logic matching project semantics
            if any(w in text_lower for w in ["clinic", "outreach", "women", "hospital", "matatu", "repair", "farm", "screened", "dewormed"]):
                # High score for first candidate label
                scores = [0.95] + [0.05 / (len(candidate_labels) - 1)] * (len(candidate_labels) - 1)
                labels = list(candidate_labels)
            else:
                # High score for last candidate label (chit-chat / irrelevant)
                scores = [0.05 / (len(candidate_labels) - 1)] * (len(candidate_labels) - 1) + [0.95]
                labels = list(reversed(candidate_labels))

            results.append({
                "sequence": text,
                "labels": labels,
                "scores": scores,
            })
        return results[0] if single else results


class TestRelevanceClassification(unittest.TestCase):

    def setUp(self):
        self.mock_classifier = DummyPipeline()
        self.sample_rows = [
            {"text": "We held a family planning outreach and served 42 women at the dispensary."},
            {"text": "yo"},
            {"text": "Sister Nancy screened 28 women for cervical cancer at Rongai Health Center."},
            {"text": "congratulations on the new baby!! see you this weekend"},
            {"text": ""},  # empty line
            {"text": "   "},  # whitespace line
        ]
        self.dataset = Dataset.from_list(self.sample_rows)
        self.dataset_dict = DatasetDict({
            "train": Dataset.from_list(self.sample_rows),
            "test": Dataset.from_list(self.sample_rows[:3]),
        })

    def test_classify_dataset_adds_relevance_column(self):
        """Test that classify_dataset_relevance adds the 'relevance' column."""
        result = classify_dataset_relevance(
            self.dataset,
            classifier=self.mock_classifier,
            relevance_col="relevance",
        )
        self.assertIn("relevance", result.column_names)
        self.assertEqual(len(result), len(self.dataset))

        # Check values
        self.assertEqual(result[0]["relevance"], "relevant field activity")
        self.assertEqual(result[1]["relevance"], "casual chit-chat")
        self.assertEqual(result[2]["relevance"], "relevant field activity")
        self.assertEqual(result[3]["relevance"], "casual chit-chat")
        # Empty lines should be marked with fallback (casual chit-chat)
        self.assertEqual(result[4]["relevance"], "casual chit-chat")
        self.assertEqual(result[5]["relevance"], "casual chit-chat")

    def test_classify_dataset_dict(self):
        """Test that classify_dataset_relevance works on a DatasetDict across all splits."""
        result = classify_dataset_relevance(
            self.dataset_dict,
            classifier=self.mock_classifier,
        )
        self.assertIsInstance(result, DatasetDict)
        self.assertIn("relevance", result["train"].column_names)
        self.assertIn("relevance", result["test"].column_names)
        self.assertEqual(result["train"][0]["relevance"], "relevant field activity")
        self.assertEqual(result["test"][0]["relevance"], "relevant field activity")

    def test_custom_candidate_labels(self):
        """Test classification with custom labels tailored for other categories."""
        custom_labels = ["Event", "Chit-chat"]
        result = classify_dataset_relevance(
            self.dataset,
            classifier=self.mock_classifier,
            candidate_labels=custom_labels,
        )
        self.assertEqual(result[0]["relevance"], "Event")
        self.assertEqual(result[1]["relevance"], "Chit-chat")

    def test_label_mapping(self):
        """Test mapping long model labels to concise output labels."""
        mapping = {
            "relevant field activity": "relevant",
            "casual chit-chat": "irrelevant",
        }
        result = classify_dataset_relevance(
            self.dataset,
            classifier=self.mock_classifier,
            label_mapping=mapping,
        )
        self.assertEqual(result[0]["relevance"], "relevant")
        self.assertEqual(result[1]["relevance"], "irrelevant")

    def test_score_col(self):
        """Test adding confidence score column."""
        result = classify_dataset_relevance(
            self.dataset,
            classifier=self.mock_classifier,
            score_col="relevance_score",
        )
        self.assertIn("relevance_score", result.column_names)
        self.assertAlmostEqual(result[0]["relevance_score"], 0.95, places=2)
        self.assertEqual(result[4]["relevance_score"], 0.0)

    def test_column_auto_detection(self):
        """Test auto-detecting transcript column when 'text' column is absent."""
        alt_dataset = Dataset.from_list([
            {"transcript": "Clinic outreach in Rongai went very well today."},
            {"transcript": "bye bye see you"},
        ])
        resolved = _resolve_text_column(alt_dataset, preferred_col="text")
        self.assertEqual(resolved, "transcript")

        result = classify_dataset_relevance(
            alt_dataset,
            classifier=self.mock_classifier,
            text_col="text",  # will auto-detect 'transcript'
        )
        self.assertIn("relevance", result.column_names)
        self.assertEqual(result[0]["relevance"], "relevant field activity")

    def test_pandas_dataframe_support(self):
        """Test passing a pandas DataFrame directly."""
        df = pd.DataFrame(self.sample_rows)
        result_df = classify_dataset_relevance(
            df,
            classifier=self.mock_classifier,
        )
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn("relevance", result_df.columns)
        self.assertEqual(result_df.iloc[0]["relevance"], "relevant field activity")

    def test_filter_relevant_lines(self):
        """Test filtering out irrelevant lines."""
        classified = classify_dataset_relevance(
            self.dataset,
            classifier=self.mock_classifier,
        )
        filtered = filter_relevant_lines(classified)
        # Rows 0 and 2 are relevant, rows 1, 3, 4, 5 are chit-chat/empty
        self.assertEqual(len(filtered), 2)
        for row in filtered:
            self.assertEqual(row["relevance"], "relevant field activity")

    def test_classify_single_line(self):
        """Test classify_line helper function."""
        label, score = classify_line(
            "We screened 28 women for cervical cancer.",
            classifier=self.mock_classifier,
        )
        self.assertEqual(label, "relevant field activity")
        self.assertAlmostEqual(score, 0.95, places=2)

        empty_label, empty_score = classify_line(
            "",
            classifier=self.mock_classifier,
        )
        self.assertEqual(empty_label, "casual chit-chat")
        self.assertEqual(empty_score, 0.0)


if __name__ == "__main__":
    unittest.main()
