from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def summarize_text(text: str, model_name: str, min_length: int, input_block_size: int, prefix: str = "Summarize: ") -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    encoded_prefix = tokenizer(prefix, return_tensors="pt").to(device)

    # Repeatedly summarize the text in blocks until it fits in one summary
    if inputs["input_ids"].shape[1] > input_block_size and input_block_size > 0:
        input_block_size -= encoded_prefix["input_ids"].shape[1]

        input_ids = inputs["input_ids"][0]
        blocks = [input_ids[i:i + input_block_size] for i in range(0, len(input_ids), input_block_size)]
        while len(blocks) > 1:
            summaries = []
            for chunk in blocks:
                chunk = chunk.unsqueeze(0)
                chunk = torch.cat([encoded_prefix["input_ids"], chunk], dim=1)

                output = model.generate(
                    chunk,
                    min_length=min_length,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.92,
                    top_k=50
                )
                
                summary_chunk = tokenizer.decode(output[0], skip_special_tokens=True)
                #print(f"\nGenerated summary: {summary_chunk}\nFrom input: {tokenizer.decode(chunk[0], skip_special_tokens=True)}")
                summaries.append(summary_chunk)
            #blocks = [*tokenizer(block_summary, return_tensors="pt", is_split_into_words=True)["input_ids"][0] for block_summary in summaries]
            new_ids = []
            for block_summary in summaries:
                block_input = tokenizer(block_summary, return_tensors="pt").to(device)["input_ids"][0]
                new_ids.extend(block_input)
            input_ids = torch.tensor(new_ids).to(device)
            blocks = [input_ids[i:i + input_block_size] for i in range(0, len(input_ids), input_block_size)]
        return tokenizer.decode(blocks[0], skip_special_tokens=True)

    inputs["input_ids"] = torch.cat([encoded_prefix["input_ids"], inputs["input_ids"]], dim=1)

    output = model.generate(**inputs, min_length=min_length,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        top_k=50
    )
    summary = tokenizer.decode(output[0], skip_special_tokens=True)

    return summary