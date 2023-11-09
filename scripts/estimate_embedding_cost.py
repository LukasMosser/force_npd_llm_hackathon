import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm
from datasets import load_dataset, Value, Features

features = Features({'doc_id': Value('string'), 'meta': Value('string'), 'raw_content': Value('string')})

# Encoder for embedding-ada-002
enc = tiktoken.get_encoding("cl100k_base")

force_llm_scrubbed_dataset = load_dataset("json", 
                                          data_files="data/force_llm_corpus_scrubbed_embedding_docs.jsonl", 
                                          split='train', 
                                          features=features)

total_tokens = 0
prg_bar = tqdm(force_llm_scrubbed_dataset)
for row in prg_bar:
    text = row['raw_content']
    try:
        tokenized = enc.encode(text)
    except TypeError:
        print(text)
        print(row)
        break
    num_tokens = len(tokenized)
    total_tokens += num_tokens
    current_cost = total_tokens/1000*0.0004
    prg_bar.set_description_str(f"Total tokens: {total_tokens}. Total Cost: ${current_cost}")