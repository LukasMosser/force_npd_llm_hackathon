import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm
from datasets import load_dataset, Value, Features

features = Features({'doc_id': Value('string'), 
                     'meta': Features({'_id': Value(dtype='string', id=None), 
                                               'corpus': Value(dtype='string', id=None),
                                                 'langdetect': Value(dtype='string', id=None), 
                                                 'possible_lanaguage': Value(dtype='string', id=None), 
                                                 'content_could_be_natural_language': Value(dtype='string', id=None)
                                                 }), 
                    'raw_content': Features({'Document Corpus': Value(dtype='string', id=None),
                                                 'Filename': Value(dtype='string', id=None),
                                                 'Content': Value('string')
                                                 })
})
# Encoder for embedding-ada-002
enc = tiktoken.get_encoding("cl100k_base")

force_llm_scrubbed_dataset = load_dataset("json", 
                                          data_files="data/force_llm_corpus_scrubbed_embedding_docs.jsonl", 
                                          split='train', 
                                          features=features)

total_tokens = 0
prg_bar = tqdm(force_llm_scrubbed_dataset)
for row in prg_bar:
    text = row['raw_content']["Content"]

    tokenized = enc.encode(text)
        
    num_tokens = len(tokenized)
    total_tokens += num_tokens
    current_cost = total_tokens/1000*0.0004
    prg_bar.set_description_str(f"Total tokens: {total_tokens}. Total Cost: ${current_cost}")