import pandas as pd
from typing import Dict
import logging 
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

CORPORA = {
    "NL_NOG_PR": "Netherlands - Netherlands Oil & Gas Portal reports.csv",
    "NO_NPD_DI": "Norway - Diskos reports.csv",
    "NO_NPD_RR": "Norway - Norwegian Petroleum Directorate relinquishment reports.csv",
    "UK_NTA_NDR": "UK - North Sea Transition Authority NDR reports.csv",
    "UK_NTA_RR": "UK - North Sea Transition Authority relinquishment reports.csv"
}

def make_jsonl_dataset(corpora: Dict[str, str], output_path: str, raw_content_column: str = 'content') -> None :
    root.info(f"Generating dataset. Corpora: {corpora}, Raw Content Column: {raw_content_column}, Output Path: {output_path}")
    df_out = pd.DataFrame()
    for corpus, fpath in corpora.items():
        # Read the CSV file with low memory to avoid dtype warning for mixed types
        df = pd.read_csv(fpath, low_memory=True)
        root.info(f"Loaded {fpath} successfully.")

        # Create the entries for the corpus including metadata
        df_temp = pd.DataFrame()
        df_temp['raw_content'] = df[raw_content_column].astype(str)
        df_temp['doc_id'] = corpus + '/' + df['filename'].astype(str) + '/' + df['page'].astype(str)
        df_temp['meta'] = '{"_id":' + df['_id'] + '"corpus": ' + corpus + ', "possible_lanaguage":' + df['possible_language'].astype(str) + ', "langdetect":' + df['langdetect'].astype(str) + '}'

        # Append to the output DataFrame
        df_out = pd.concat([df_out, df_temp], ignore_index=True)
        root.info(f"Concatenated to corpus")
    
    root.info("Writing corpus to disk.")
    df_out.to_json(output_path, orient="records", lines=True)
    root.info(f"Successfully wrote corpus to disk at: {output_path}")

if __name__ == "__main__":

    input_directory = "./data/raw"
    output_directory = "./data"

    corpora = {corpus: input_directory+'/'+path for corpus, path in CORPORA.items()}

    make_jsonl_dataset(corpora=corpora, 
                       output_path=f"{output_directory}/force_llm_corpus_raw.jsonl", 
                       raw_content_column='content')
    
    make_jsonl_dataset(corpora=corpora, 
                    output_path=f"{output_directory}/force_llm_corpus_scrubbed.jsonl", 
                    raw_content_column='content_scrubbed_light')
