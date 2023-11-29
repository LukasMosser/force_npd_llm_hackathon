# FORCE NPD LLM Hackathon

This repository contains some materials for getting started in the FORCE Language Modeling Hackathon, 29-30 November and 1 December 2023.

[Find out more or sign up here.](https://www.npd.no/en/force/events/in-person--the-npd-language-modelling-hackathon-2023/)

## Introduction

Large Language Models or LLMs have gained enormous amounts of attention with the advent of ChatGPT. Numerous use cases across various industries have shown their potential as generalist interfaces for various tasks.  

While LLMs have shown the ability to solve complex language based tasks such as summarization, translation, semantic search and retrieval, entity recognition, or question answering, applying them to a specific scientific or engineering domain comes with many challenges.  

Off the shelf LLMs are trained with 1000s of GPU months to years in compute resources on trillions of tokens (parts of a word) with highly specialized and curated datasets, machine learning know-how, and a whole bunch of alchemical engineering tricks.  

Natural language documents in the energy industry come in many forms and many are difficult to read, contain tables and/or figures, come in various languages, use domain specific acronyms or have handwriting which must first undergo OCR before being useful, and many more challenges. Hence creating such trillion token datasets may simply not be feasible. But is it even necessary?  

## Dataset

In this hackathon, a dataset has been curated by [Fabriq](https://npd.fabriqai.com/) that comprises data from three national repositories; the Norwegian Petroleum Directorate, the UK North Sea Transition Authority, and the Dutch Oil & Gas Reports.  

The dataset is provided courtesy of FORCE, FORCE Sponsors, and the Norwegian Petroleum Directorate (NPD). The dataset is licensed under the [Norwegian Licence for Open Government Data (NLOD) 2.0](https://data.norge.no/nlod/en/2.0).

We process the dataset first to concatenate into one format that follows the standard used for very large datasets such as RedPyjama or ThePile. Number of tokens (scrubbed - cl100k_base): 755829106.

The dataset is provided as comma seperated values with a raw extracted form, and a slightly scrubbed version that has been cleaned to remove some of the most obvious issues with the datasets. This dataset forms the basis of the hackathon and is the main center of attention.  

## Introductory notebook

A starter notebook can be found in the `notebooks` folder. It contains examples for loading the dataset, preprocessing, and training a CBOW model, as well as examples for using the embeddings to compute similarity and perform retrieval augmented generation.
In addition there is also an example of fine-tuning your own sentence-level embedding model. 

In the notebook...

1. We will explore the dataset, perform some simple analytics, and turn it into a format that is easily used for downstream machine learning tasks.
2. We will preprocess the dataset and normalize it so its content can be readily ingested into LLMs
3. We will train a continuous bag of words model (CBOW) - a word level embedding model to exemplify getting domain specific word level embeddings.
4. Going from individual words we will expand to sentences and use pre-trained models powered by Azure OpenAI to identify relationships.
5. We will then attempt to build our own sentence level embedding model and overcome the cold-start problem using OpenAIs GPT4 and a small transformer model.

## Preprocessing

Examples for preprocessing the dataset are provided in the `scripts` folder.

- `scripts/make_dataset.py` creates one big jsonl file from the raw dataset with metadata stored per entry.
- `scripts/make_embedding_docs.py` creates a chunked version of the dataset that can be used for computing embeddings and subsequently retrieval augmented generation. 
- `scripts/estimate_embedding_cost.py` estimates the total cost of producing embeddings from the chunked embeddings dataset.
- `scripts/compute_embeddings.py` example of how the ada2 embeddings were generated
- `scripts/filter_embeddings_natural_language.py` example how the whole embeddings were trimmed down to only the natural language embeddings
- `scripts/read_embeddings_file.py` part of the embeddings file is corrupted hence it must be read character by character. This shows how.

## Installation

To install the required dependencies please create a virtual environment and install the dependencies from the requirements file:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

We have two .env files, one for shared variables that can be distributed openly, and one for the API key which will be distributed as part of the hackathon. 
`.env.shared` contains the variables we will use in the notebooks and scripts. 
`.env.secret` contains the API key for the Azure OpenAI API.

## License

The dataset is licensed under the [Norwegian Licence for Open Government Data (NLOD) 2.0](https://data.norge.no/nlod/en/2.0).

Code in this repository is licensed under the [MIT license](https://opensource.org/license/mit/).
