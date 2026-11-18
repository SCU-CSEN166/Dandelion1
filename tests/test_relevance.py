"""Unit tests for the relevance classification stage."""
import sys
import os
import unittest
from datasets import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from relevance import EVENT_DESCRIPTION, EVENT_THRESHOLD, classify_dataset_relevance, is_chitchat


class DummyPipeline:
    """Mock zero-shot pipeline: a high Event score for messages with activity words, a low one otherwise."""

    def __init__(self, event_score=0.99, other_score=0.05):
        self.event_score, self.other_score = event_score, other_score
        self.calls = []

    def __call__(self, inputs, candidate_labels=None, hypothesis_template=None, multi_label=False, batch_size=None):
        self.calls.append({"candidate_labels": candidate_labels, "multi_label": multi_label})
        single = isinstance(inputs, str)
        results = []
        for text in [inputs] if single else inputs:
            activity = any(w in text.lower() for w in ["clinic", "outreach", "women", "screened", "dewormed"])
            results.append({"sequence": text, "labels": list(candidate_labels),
                            "scores": [self.event_score if activity else self.other_score]})
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

    def test_labels_event_and_chitchat(self):
        result = classify_dataset_relevance(self.dataset, classifier=self.mock_classifier)
        self.assertEqual(result["relevance"], ["Event", "Chit-chat", "Event", "Chit-chat", "Chit-chat", "Chit-chat"])
        # one yes/no question per message: does it entail the Event description?
        self.assertEqual(self.mock_classifier.calls[0], {"candidate_labels": [EVENT_DESCRIPTION], "multi_label": True})

    def test_threshold(self):
        """An Event score below the threshold is Chit-chat, at or above it is Event."""
        for score, label in ((EVENT_THRESHOLD - 0.01, "Chit-chat"), (EVENT_THRESHOLD, "Event")):
            result = classify_dataset_relevance(self.dataset, classifier=DummyPipeline(event_score=score))
            self.assertEqual(result[0]["relevance"], label)

    def test_score_col(self):
        result = classify_dataset_relevance(self.dataset, classifier=self.mock_classifier, score_col="relevance_score")
        self.assertAlmostEqual(result[0]["relevance_score"], 0.99, places=2)
        self.assertAlmostEqual(result[1]["relevance_score"], 0.05, places=2)
        self.assertEqual(result[4]["relevance_score"], 0.0)  # blank lines are not sent to the model

    def test_custom_text_column(self):
        alt_dataset = Dataset.from_list([
            {"transcript": "Clinic outreach in Rongai went very well today."},
            {"transcript": "bye bye see you"},
        ])
        result = classify_dataset_relevance(alt_dataset, classifier=self.mock_classifier, text_col="transcript")
        self.assertEqual(result["relevance"], ["Event", "Chit-chat"])

    def test_is_chitchat(self):
        self.assertTrue(is_chitchat("Chit-chat"))
        self.assertFalse(is_chitchat("Event"))


if __name__ == "__main__":
    unittest.main()
