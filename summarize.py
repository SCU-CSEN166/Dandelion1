from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def summarize_text(text: str, model_name: str, min_length) -> str:
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(**inputs, length_penalty=2.0, forced_bos_token_id=0, num_beams=4, min_length=min_length)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary