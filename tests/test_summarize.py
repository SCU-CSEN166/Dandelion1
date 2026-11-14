"""Unit tests for the summarization prompt formatting and utility functions."""
import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from summarize import format_prompt, DEFAULT_SYSTEM_PROMPT


class DummyTokenizerWithChatTemplate:
    def __init__(self):
        self.chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        res = []
        for msg in messages:
            res.append(f"<|{msg['role']}|>{msg['content']}")
        if add_generation_prompt:
            res.append("<|assistant|>")
        return "".join(res)


class DummyTokenizerWithoutTemplate:
    chat_template = None


class TestSummarize(unittest.TestCase):

    def test_format_prompt_encoder_decoder(self):
        tok = DummyTokenizerWithoutTemplate()
        text = "Person0: Hello team\nPerson1: Outreach was successful."
        prompt = format_prompt(tok, text, is_encoder_decoder=True)

        self.assertIn("Chat Transcript:", prompt)
        self.assertIn(text, prompt)
        self.assertIn("Field Outcomes Report:", prompt)
        self.assertIn(DEFAULT_SYSTEM_PROMPT, prompt)

    def test_format_prompt_custom_prefix(self):
        tok = DummyTokenizerWithoutTemplate()
        text = "Outreach took place."
        prompt = format_prompt(tok, text, is_encoder_decoder=True, custom_prefix="summarize: ")
        self.assertEqual(prompt, "summarize: Outreach took place.")

    def test_format_prompt_chat_template(self):
        tok = DummyTokenizerWithChatTemplate()
        text = "Outreach took place."
        prompt = format_prompt(tok, text, is_encoder_decoder=False)

        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertIn("<|assistant|>", prompt)
        self.assertIn("Here is the chat transcript:", prompt)

    def test_format_prompt_with_extracted_info_list(self):
        tok = DummyTokenizerWithoutTemplate()
        text = "Outreach took place."
        extracted = ["- Event: Family planning | Attendees: 42", "- Event: Cervical screening | Attendees: 28"]
        prompt = format_prompt(tok, text, is_encoder_decoder=True, extracted_info=extracted)

        self.assertIn("Extracted Field Details:", prompt)
        self.assertIn("Family planning | Attendees: 42", prompt)
        self.assertIn("Cervical screening | Attendees: 28", prompt)
        self.assertIn(text, prompt)

    def test_format_prompt_with_extracted_info_str(self):
        tok = DummyTokenizerWithoutTemplate()
        text = "Outreach took place."
        extracted = "- Event: Table banking day"
        prompt = format_prompt(tok, text, is_encoder_decoder=True, extracted_info=extracted)

        self.assertIn("Extracted Field Details:", prompt)
        self.assertIn("- Event: Table banking day", prompt)


if __name__ == "__main__":
    unittest.main()
