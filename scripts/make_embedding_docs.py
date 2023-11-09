import pandas as pd
from typing import Dict
import logging 
import sys
from tqdm.auto import tqdm 

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

def split_text_into_chunks_for_loop(text, chunk_size, overlap_fraction):
    """
    Split a text string into overlapping chunks of words using a for loop.

    :param text: The text to split.
    :param chunk_size: The number of words per chunk.
    :param overlap_fraction: The fraction of overlap between chunks.
    :return: A list of text chunks.
    """
    # Validate parametersß
    if not 0 <= overlap_fraction < 1:
        raise ValueError("Overlap fraction must be between 0 and 1, non-inclusive.")

    if chunk_size < 1:
        raise ValueError("Chunk size must be greater than 0.")

    words = text.split()
    chunks = []
    overlap_size = int(chunk_size * overlap_fraction)

    # Validate overlap size
    if overlap_size >= chunk_size:
        raise ValueError("Overlap size must be smaller than chunk size.")

    # Calculate the number of chunks
    num_chunks = ((len(words) - overlap_size) // (chunk_size - overlap_size)) + 1 if len(words) >= chunk_size else 1

    for i in range(0, num_chunks):
        # Calculate the start index for each chunk
        start_index = i * (chunk_size - overlap_size)
        end_index = start_index + chunk_size
        # Create a chunk and append to the list
        chunks.append(' '.join(words[start_index:end_index]))

    return chunks


def make_row_data(corpus, row, chunk_size, overlap, content_column):
    filename = str(row['filename'])
    chunks = split_text_into_chunks_for_loop(str(row[content_column]), chunk_size=chunk_size, overlap_fraction=overlap)

    for idx, chunk in enumerate(chunks):
        chunk_dict = {'raw_content': str('{'+f'"Content": "{chunk}", "Document Corpus": "{corpus}", "Filename": "{filename}"'+'}'),
                'doc_id': str(corpus + '/' + str(row['filename']) + '/' + str(row['page'])+'/'+str(idx)), 
                'meta': str('{"_id":' + row['_id'] + '"corpus": ' + corpus + ', "possible_lanaguage":' + str(row['possible_language']) + ', "langdetect":' + str(row['langdetect']) + '}')
                }
        yield chunk_dict

def make_jsonl_dataset(corpora: Dict[str, str], output_path: str, raw_content_column: str = 'content', chunk_size: int = 500, overlap: float = 0.66) -> None :
    root.info(f"Generating dataset. Corpora: {corpora}, Raw Content Column: {raw_content_column}, Output Path: {output_path}")
    df_out = pd.DataFrame()
    for corpus, fpath in corpora.items():
        # Read the CSV file with low memory to avoid dtype warning for mixed types
        df = pd.read_csv(fpath, low_memory=True)
        root.info(f"Loaded {fpath} successfully.")

        # Create the entries for the corpus including metadata
        data = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            for chunk_dict in make_row_data(corpus, row, chunk_size, overlap, raw_content_column):
                data.append(chunk_dict)
        df_temp = pd.DataFrame(data, columns=["doc_id", "raw_content", "meta"], dtype=str)
        # Append to the output DataFrame
        df_out = pd.concat([df_out, df_temp], ignore_index=True)
        root.info(f"Concatenated to corpus")
    
    root.info("Writing corpus to disk.")
    df_out.to_json(output_path, orient="records", lines=True)
    root.info(f"Successfully wrote corpus to disk at: {output_path}")

if __name__ == "__main__":

    input_directory = "./data/raw"
    output_directory = "./data"
    chunk_size = 300
    overlap = 0.66

    corpora = {corpus: input_directory+'/'+path for corpus, path in CORPORA.items()}
    
    make_jsonl_dataset(corpora=corpora, 
                       output_path=f"{output_directory}/force_llm_corpus_scrubbed_embedding_docs.jsonl", 
                       raw_content_column='content_scrubbed_light', 
                       chunk_size=chunk_size,
                       overlap=overlap
                    )
