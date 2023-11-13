from openai import AzureOpenAI
from openai.types import Embedding
from dotenv import load_dotenv
import os
from datasets import load_dataset, ClassLabel, Value, Features
import json
from tqdm.auto import tqdm 

features = Features({'doc_id': Value('string'), 'meta': Value('string'), 'raw_content': Value('string')})

force_llm_dataset_scrubbed = load_dataset("json", data_files="./data/force_llm_corpus_scrubbed.jsonl", features=features)

def fetch_dataset_records(dataset, batch_size=16):
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        yield dataset[start_idx:end_idx]

# Load the shared environment variables, not secrets
load_dotenv(".env.shared")
load_dotenv(".env.secret")

openai_client = AzureOpenAI(
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"]
)

embeddings_list = []
batch_size = 16	
with open("./data/embeddings.jsonl", "w+") as f:
    for batch in tqdm(fetch_dataset_records(force_llm_dataset_scrubbed['train'], batch_size=batch_size), total=len(force_llm_dataset_scrubbed['train']) // batch_size):
        try:
            embeddings = openai_client.embeddings.create(input=batch["raw_content"], 
                                        model=os.environ["ADA002_DEPLOYMENT"])
            print(batch)
            for batch_element, embedding in zip(batch["doc_id"], embeddings.data):
                f.write(json.dumps({"doc_id": batch_element, "embedding": embedding.embedding}) + "\n")

        except Exception as e:
            print(e)
            for i, id, text in enumerate(zip(batch["doc_id"], batch["raw_content"])):
                try:
                    embedding = openai_client.embeddings.create(input=text, 
                                        model=os.environ["ADA002_DEPLOYMENT"])
                    f.write(json.dumps({"doc_id": id, "embedding": embedding.data[0].embedding}) + "\n")
                except Exception as e:
                    f.write(json.dumps({"doc_id": id, "embedding": []}) + "\n")
