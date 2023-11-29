from datasets import load_dataset, Value, Features, Sequence
import gzip
import json
from tqdm.auto import tqdm
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

force_llm_dataset_scrubbed = load_dataset("json", data_files="./data/force_llm_corpus_scrubbed_embedding_docs.jsonl", features=features).filter(lambda x: x['meta']["content_could_be_natural_language"] == "True")

print(len(force_llm_dataset_scrubbed["train"]))

doc_ids = [row["doc_id"]for row in force_llm_dataset_scrubbed["train"]]

print(doc_ids[:10])

force_llm_dataset_scrubbed_embeddings = load_dataset("json", 
                                          data_files="./data/force_llm_corpus_scrubbed_embeddings.jsonl", 
                                          streaming=True, 
                                          features=Features({'doc_id': Value('string'), 'embedding': Sequence(Value("float32")) })
                                          ).filter(lambda x: x['doc_id'] in doc_ids)


with gzip.open("./data/force_llm_corpus_scrubbed_embeddings_natural.jsonl.gz", 'w') as fout:
    prg = tqdm(force_llm_dataset_scrubbed_embeddings["train"])
    for row in prg:
        data = {"doc_id": row["doc_id"], "embedding": row["embedding"]}
        fout.write(json.dumps(data).encode('utf-8'))     
        prg.set_description_str(f"Writing {row['doc_id']}")                  