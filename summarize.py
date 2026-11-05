from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def summarize_text(text: str, model_name: str, min_length) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    output = model.generate(**inputs, length_penalty=2.0, forced_bos_token_id=0, num_beams=4, min_length=min_length, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)

    return summary