import json
from tqdm.auto import tqdm
# Function to parse a JSONL file where each line is a JSON object.
# Invalid JSON lines are skipped.

def parse_jsonl_char_by_char(file_path):
    valid_items = []
    invalid_count = 0
    buffer = ""

    with open(file_path, 'r', encoding='utf-8') as file:
        prg_bar = tqdm()
        for char in iter(lambda: file.read(1), ''):
            buffer += char
            # Try to parse when we encounter a newline character
            if buffer[-1] == '}':
                try:
                    item = json.loads(buffer)
                    valid_items.append(item)
                    prg_bar.update(1)
                except json.JSONDecodeError:
                    invalid_count += 1
                # Reset the buffer for the next line
                buffer = ""

    return valid_items, invalid_count

valid_items, _ = parse_jsonl_char_by_char("data/force_llm_corpus_scrubbed_embeddings_natural.jsonl")
# This function can be used to process the JSONL file line by line,
# effectively handling each JSON object and skipping those that are invalid.
