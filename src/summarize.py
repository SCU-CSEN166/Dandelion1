from functools import lru_cache
import gc
import re
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from preprocess import clean_chat_text

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

def format_prompt(
    tokenizer,
    text: str,
    is_encoder_decoder: bool,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    custom_prefix: str | None = None,
    extracted_info: list[str] | str | None = None,
) -> str:
    """Format input text with chat template if supported by model, otherwise fall back to structured text."""
    if custom_prefix is not None:
        return f"{custom_prefix}{text}"

    # Prepare extracted facts context if provided by the extraction stage
    context_str = ""
    if extracted_info:
        if isinstance(extracted_info, list):
            context_str = "Extracted Field Details:\n" + "\n".join(extracted_info) + "\n\n"
        elif str(extracted_info).strip():
            context_str = f"Extracted Field Details:\n{str(extracted_info).strip()}\n\n"

    transcript_content = f"{context_str}Here is the chat transcript:\n\n{text}"

    if not is_encoder_decoder and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
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
    model_kwargs = {"low_cpu_mem_usage": True}

    if torch.cuda.is_available():
        model_kwargs["dtype"] = (
            torch.bfloat16
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            else "auto"
        )
        try:
            import accelerate
            model_kwargs["device_map"] = "auto"
        except ImportError:
            pass

    try:
        config = AutoConfig.from_pretrained(model_name)
        is_enc_dec = getattr(config, "is_encoder_decoder", False)
    except Exception:
        is_enc_dec = False

    loader = AutoModelForSeq2SeqLM if is_enc_dec else AutoModelForCausalLM
    try:
        model = loader.from_pretrained(model_name, **model_kwargs)
    except Exception:
        fallback = AutoModelForCausalLM if is_enc_dec else AutoModelForSeq2SeqLM
        model = fallback.from_pretrained(model_name, **model_kwargs)

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
    """Generate summary text for a single input tensor with OOM fallback handling."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    is_enc_dec = getattr(model.config, "is_encoder_decoder", False)

    kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.15,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        fallback = dict(kwargs, do_sample=False, max_new_tokens=max(32, max_new_tokens // 2))
        fallback.pop("temperature", None)
        fallback.pop("top_p", None)
        fallback.pop("top_k", None)
        if "min_new_tokens" in fallback:
            fallback["min_new_tokens"] = min(fallback["min_new_tokens"], fallback["max_new_tokens"])
        try:
            output = _run(fallback)
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            input_ids = input_ids[:, :max(16, input_ids.shape[-1] // 2)]
            fallback["attention_mask"] = torch.ones_like(input_ids, device=device)
            fallback["max_new_tokens"] = max(16, fallback["max_new_tokens"] // 2)
            if "min_new_tokens" in fallback:
                fallback["min_new_tokens"] = min(fallback["min_new_tokens"], fallback["max_new_tokens"])
            output = _run(fallback)

    if is_enc_dec:
        summary_ids = output[0]
    else:
        prompt_len = input_ids.shape[-1]
        summary_ids = output[0][prompt_len:]
        if len(summary_ids) == 0:
            summary_ids = output[0]

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
    split_overflowing: bool | int = False,
    prefix: str | None = None,
    max_new_tokens: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    clean_chat: bool = True,
    extracted_info: list[str] | str | None = None,
    **kwargs
) -> str:
    """Summarize chat text into a structured field report."""
    tokenizer, model = load_model(model_name)
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    is_enc_dec = getattr(model.config, "is_encoder_decoder", False)

    if clean_chat:
        text = clean_chat_text(text)
        if not text:
            return "No relevant dialogue found to summarize."

    model_max_len = get_safe_max_length(model, tokenizer)
    if isinstance(split_overflowing, int) and not isinstance(split_overflowing, bool):
        use_blocks = split_overflowing > 0
        block_size = min(split_overflowing, model_max_len) if split_overflowing > 0 else model_max_len
    else:
        use_blocks = bool(split_overflowing)
        block_size = model_max_len

    if max_new_tokens is None:
        max_new_tokens = max(128, min_length + 64)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _generate(prompt_str: str) -> str:
        ids = tokenizer(
            prompt_str,
            return_tensors="pt",
            truncation=True,
            max_length=block_size,
            add_special_tokens=True
        )["input_ids"]
        return _generate_chunk(model, tokenizer, ids, min_length, max_new_tokens, device, kwargs)

    # Check if input fits within block_size
    formatted_prompt = format_prompt(tokenizer, text, is_enc_dec, system_prompt, prefix, extracted_info=extracted_info)
    prompt_len = len(tokenizer.encode(formatted_prompt, add_special_tokens=True))

    if not use_blocks or prompt_len <= block_size:
        return _generate(formatted_prompt)

    # Hierarchical summarization for long inputs
    base_len = len(tokenizer.encode(format_prompt(tokenizer, "", is_enc_dec, system_prompt, prefix), add_special_tokens=False))
    max_chunk_tokens = max(64, block_size - base_len - 32)

    for _ in range(10):
        chunks = tokenizer(
            text,
            truncation=True,
            max_length=max_chunk_tokens,
            return_overflowing_tokens=True,
            add_special_tokens=False
        )["input_ids"]

        if len(chunks) <= 1:
            break

        summaries = []
        for chunk in chunks:
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            chunk_summary = _generate(format_prompt(tokenizer, chunk_text, is_enc_dec, system_prompt, prefix))
            if chunk_summary:
                summaries.append(chunk_summary)

        new_text = "\n\n".join(summaries)
        if len(new_text) >= len(text) or len(summaries) <= 1:
            text = new_text
            break
        text = new_text

    return _generate(format_prompt(tokenizer, text, is_enc_dec, system_prompt, prefix, extracted_info=extracted_info))
