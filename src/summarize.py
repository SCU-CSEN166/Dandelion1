from functools import lru_cache
import gc
import re
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_SYSTEM_PROMPT = (
    "You are an operations analyst summarizing field communications for Dandelion Africa.\n"
    "Analyze the provided redacted WhatsApp conversation and produce a concise, structured Field Outcomes Report:\n"
    "1. Practical Actions Completed: (Concrete tasks finished in the field)\n"
    "2. Blockers & Technical Issues: (Equipment, vehicle, or resource problems mentioned)\n"
    "3. Outcomes & Resolutions: (Decisions made, workarounds found, or agreements reached)\n"
    "4. Pending Follow-ups: (Unresolved items or next steps)\n\n"
    "Note: Identifiers like Person0, Person1 represent anonymized field personnel. "
    "Focus strictly on actionable field outcomes and facts; ignore casual banter."
)

# For encoder-decoder models that declare no summarization setup of their own (e.g. Flan-T5)
ENC_DEC_INSTRUCTION = "Summarize the following conversation:\n"


def encoder_decoder_prefix(config) -> str:
    """What goes in front of the transcript for an encoder-decoder model.

    These models (BART, T5, Pegasus...) are not chat models: a long system prompt is just more text to summarize.
    A model that declares a summarization setup gets exactly its own prefix (T5: "summarize: ", BART-CNN: none);
    any other gets a one-line instruction.
    """
    params = (getattr(config, "task_specific_params", None) or {}).get("summarization")
    return params.get("prefix", "") if params is not None else ENC_DEC_INSTRUCTION


def format_prompt(
    tokenizer,
    text: str,
    is_encoder_decoder: bool,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    custom_prefix: str | None = None,
    extracted_info: list[str] | str | None = None,
    encoder_decoder_prefix: str = ENC_DEC_INSTRUCTION,
) -> str:
    """Build the model input.

    - custom_prefix: the caller's own prefix + text.
    - Encoder-decoder models: encoder_decoder_prefix + text only (no system prompt, no extracted facts).
    - Chat models: system prompt, extracted facts and transcript in the model's chat template.
    - Other decoder-only models: the same content as plain text, ending in a cue to start the report.
    """
    if custom_prefix is not None:
        return f"{custom_prefix}{text}"
    if is_encoder_decoder:
        return f"{encoder_decoder_prefix}{text}"

    # Prepare extracted facts context if provided by the extraction stage
    context_str = ""
    if extracted_info:
        if isinstance(extracted_info, list):
            context_str = "Extracted Field Details:\n" + "\n".join(extracted_info) + "\n\n"
        elif str(extracted_info).strip():
            context_str = f"Extracted Field Details:\n{str(extracted_info).strip()}\n\n"

    transcript_content = f"{context_str}Here is the chat transcript:\n\n{text}"

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript_content},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Handle chat templates that disallow system role
            try:
                user_content = f"{system_prompt}\n\n{transcript_content}"
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

    return f"{system_prompt}\n\n{context_str}Chat Transcript:\n{text}\n\nField Outcomes Report:\n"


@lru_cache(maxsize=2)
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else "auto"
        try:
            import accelerate  # noqa: F401  (optional: spreads large models over the available GPUs)
            model_kwargs["device_map"] = "auto"
        except ImportError:
            pass

    is_enc_dec = getattr(AutoConfig.from_pretrained(model_name), "is_encoder_decoder", False)
    loader = AutoModelForSeq2SeqLM if is_enc_dec else AutoModelForCausalLM
    model = loader.from_pretrained(model_name, **model_kwargs)

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if "device_map" not in model_kwargs:
        model = model.to(device)

    model.eval()
    return tokenizer, model


def get_safe_max_length(model, tokenizer, cap: int = 4096) -> int:
    """Determine a safe max sequence length for the model to prevent OOM."""
    candidates = []
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_sequence_length"):
            val = getattr(config, attr, None)
            if isinstance(val, int) and 0 < val < 1000000:
                candidates.append(val)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 1000000:
        candidates.append(tok_max)

    return min(min(candidates), cap) if candidates else 1024


def _generate_chunk(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    min_length: int,
    max_new_tokens: int,
    device: torch.device,
    generation_kwargs: dict | None = None
) -> str:
    """Generate summary text for a single input tensor (on out-of-memory, retry once with half the new tokens).

    Returns "" when the model generates nothing.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    is_enc_dec = getattr(model.config, "is_encoder_decoder", False)

    kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if not is_enc_dec:
        # Encoder-decoder summarizers ship their own decoding settings (BART-CNN: 4 beams, no repeated 3-grams);
        # decoder-only models get a mild repetition penalty instead
        kwargs["repetition_penalty"] = 1.15
    if generation_kwargs:
        kwargs.update(generation_kwargs)

    if min_length > 0:
        if is_enc_dec:
            kwargs["min_length"] = min_length
        else:
            kwargs["min_new_tokens"] = min(min_length, max_new_tokens)

    def _run(args):
        with torch.inference_mode():
            return model.generate(input_ids, **args)

    try:
        output = _run(kwargs)
    except torch.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        fallback = dict(kwargs, max_new_tokens=max(32, max_new_tokens // 2))
        if "min_new_tokens" in fallback:
            fallback["min_new_tokens"] = min(fallback["min_new_tokens"], fallback["max_new_tokens"])
        output = _run(fallback)

    # A decoder-only model's output starts with the prompt: keep only the new tokens (possibly none)
    summary_ids = output[0] if is_enc_dec else output[0][input_ids.shape[-1]:]
    summary = tokenizer.decode(summary_ids, skip_special_tokens=True).strip()
    summary = re.sub(r"^<think>.*?</think>\s*", "", summary, flags=re.DOTALL).strip()

    del input_ids, attention_mask, output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def summarize_text(
    text: str,
    model_name: str,
    min_length: int = 10,
    max_input_tokens: int | None = None,
    prefix: str | None = None,
    max_new_tokens: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    extracted_info: list[str] | str | None = None,
    **kwargs
) -> str:
    """Summarize chat text (already cleaned by preprocess.py) into a structured field report.

    Input too long for the model is summarized in blocks, then the block summaries are summarized, until the
    prompt fits. max_input_tokens optionally caps the prompt length below the model's own limit.
    Returns "" if the model generates nothing.
    """
    if not text or not text.strip():
        return "No relevant dialogue found to summarize."
    tokenizer, model = load_model(model_name)
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    is_enc_dec = getattr(model.config, "is_encoder_decoder", False)
    enc_dec_prefix = encoder_decoder_prefix(model.config)

    if max_new_tokens is None:
        max_new_tokens = max(128, min_length + 64)

    block_size = get_safe_max_length(model, tokenizer)
    if max_input_tokens:
        block_size = min(block_size, max_input_tokens)
    if not is_enc_dec:  # a decoder-only model's prompt and summary share one context window
        block_size = max(256, block_size - max_new_tokens)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _format(t: str, info=None) -> str:
        return format_prompt(tokenizer, t, is_enc_dec, system_prompt, prefix, extracted_info=info,
                             encoder_decoder_prefix=enc_dec_prefix)

    def _n_tokens(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=True))

    def _generate(prompt_str: str) -> str:
        # truncation is only a safety net: the prompt has been made to fit below
        ids = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=block_size)["input_ids"]
        return _generate_chunk(model, tokenizer, ids, min_length, max_new_tokens, device, kwargs)

    # Long input: summarize it block by block, then summarize those summaries, until the whole prompt fits
    block_tokens = max(64, block_size - _n_tokens(_format("")) - 32)
    for _ in range(10):
        if _n_tokens(_format(text, extracted_info)) <= block_size:
            break
        blocks = tokenizer(text, truncation=True, max_length=block_tokens, return_overflowing_tokens=True,
                           add_special_tokens=False)["input_ids"]
        if len(blocks) <= 1:
            break
        summaries = [s for s in (_generate(_format(tokenizer.decode(b, skip_special_tokens=True))) for b in blocks)
                     if s]
        new_text = "\n\n".join(summaries)
        if not summaries or len(new_text) >= len(text):
            break  # the block summaries are no longer getting shorter
        text = new_text

    # Last resort: shorten the transcript (never the instructions or the cue to start writing) so the prompt fits
    if _n_tokens(_format(text, extracted_info)) > block_size:
        budget = max(16, block_size - _n_tokens(_format("", extracted_info)) - 8)
        text = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:budget], skip_special_tokens=True)
        print(f"Warning: input still too long after block summarization; kept its first {budget} tokens")

    return _generate(_format(text, extracted_info))
