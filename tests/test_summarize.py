"""Unit tests for the summarization prompt formatting and utility functions."""
import sys
import os
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from summarize import ENC_DEC_INSTRUCTION, DEFAULT_SYSTEM_PROMPT, _generate_chunk, encoder_decoder_prefix, format_prompt


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
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids)


class EchoModel:
    """A decoder-only 'model' that generates nothing: its output is just the prompt."""
    config = SimpleNamespace(is_encoder_decoder=False)

    def generate(self, input_ids, **kwargs):
        return input_ids


class TestSummarize(unittest.TestCase):

    def test_format_prompt_encoder_decoder(self):
        """Encoder-decoder models get the transcript only: instructions would be summarized as text."""
        tok = DummyTokenizerWithoutTemplate()
        text = "Person0: Hello team\nPerson1: Outreach was successful."
        self.assertEqual(format_prompt(tok, text, is_encoder_decoder=True, encoder_decoder_prefix=""), text)
        prompt = format_prompt(tok, text, is_encoder_decoder=True, extracted_info=["- Event: Outreach"])
        self.assertEqual(prompt, ENC_DEC_INSTRUCTION + text)
        self.assertNotIn(DEFAULT_SYSTEM_PROMPT, prompt)

    def test_encoder_decoder_prefix(self):
        bart = SimpleNamespace(task_specific_params={"summarization": {"num_beams": 4}})
        t5 = SimpleNamespace(task_specific_params={"summarization": {"prefix": "summarize: "}})
        flan = SimpleNamespace(task_specific_params=None)
        self.assertEqual(encoder_decoder_prefix(bart), "")
        self.assertEqual(encoder_decoder_prefix(t5), "summarize: ")
        self.assertEqual(encoder_decoder_prefix(flan), ENC_DEC_INSTRUCTION)

    def test_format_prompt_custom_prefix(self):
        tok = DummyTokenizerWithoutTemplate()
        text = "Outreach took place."
        prompt = format_prompt(tok, text, is_encoder_decoder=True, custom_prefix="summarize: ")
        self.assertEqual(prompt, "summarize: Outreach took place.")

    def test_format_prompt_plain_decoder(self):
        tok = DummyTokenizerWithoutTemplate()
        text = "Person0: Hello team\nPerson1: Outreach was successful."
        prompt = format_prompt(tok, text, is_encoder_decoder=False)
        self.assertIn(DEFAULT_SYSTEM_PROMPT, prompt)
        self.assertIn(text, prompt)
        self.assertTrue(prompt.endswith("Field Outcomes Report:\n"))

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
        prompt = format_prompt(tok, text, is_encoder_decoder=False, extracted_info=extracted)

        self.assertIn("Extracted Field Details:", prompt)
        self.assertIn("Family planning | Attendees: 42", prompt)
        self.assertIn("Cervical screening | Attendees: 28", prompt)
        self.assertIn(text, prompt)

    def test_format_prompt_with_extracted_info_str(self):
        tok = DummyTokenizerWithoutTemplate()
        text = "Outreach took place."
        extracted = "- Event: Table banking day"
        prompt = format_prompt(tok, text, is_encoder_decoder=False, extracted_info=extracted)

        self.assertIn("Extracted Field Details:", prompt)
        self.assertIn("- Event: Table banking day", prompt)

    def test_empty_generation_returns_empty_string(self):
        """When a decoder-only model generates no new tokens, the summary is empty, not the prompt."""
        summary = _generate_chunk(EchoModel(), DummyTokenizerWithoutTemplate(), torch.tensor([5, 6, 7]),
                                  min_length=0, max_new_tokens=16, device=torch.device("cpu"))
        self.assertEqual(summary, "")


if __name__ == "__main__":
    unittest.main()
