# FORCE NPD LLM Hackathon November 2023 Starter Pack

## Introduction
Large Language Models or LLMs have gained enormous amounts of attention with the advent of ChatGPT. Numerous use cases across various industries have shown their potential as generalist interfaces for various tasks.  

While LLMs have shown the ability to solve complex language based tasks such as summarization, translation, semantic search and retrieval, entity recognition, or question answering, applying them to a specific scientific or engineering domain comes with many challenges.  

Off the shelf LLMs are trained with 1000s of GPU months to years in compute resources on trillions of tokens (parts of a word) with highly specialized and curated datasets, machine learning know-how, and a whole bunch of alchemical engineering tricks.  

Natural language documents in the energy industry come in many forms and many are difficult to read, contain tables and/or figures, come in various languages, use domain specific acronyms or have handwriting which must first undergo OCR before being useful, and many more challenges. Hence creating such trillion token datasets may simply not be feasible. But is it even necessary?  

In this hackathon, a dataset has been curated by [Fabric]() that composes of data from three national repositories; the Norwegian Petroleum Directorate, the UK North Sea Transition Authority, and the Dutch Oil & Gas Reports.  

The dataset is provided as comma seperated values with a raw extracted form, and a slightly scrubbed version that has been cleaned to remove some of the most obvious issues with the datasets. This dataset forms the basis of the hackathon and is the main center of attention.  

To benefit from text-based technologies, we need good datasets, yet we do not have such high-quality datasets. Some companies may invest in creating their own - yet as we see in the world of open source, most progress in the world of machine learning and LLMs comes from sharing the know-how, the data, and the models that allow us to build new technology. Hence we emphasize the importance of generating such a dataset for the energy industry.  

The dataset we have created is a first attempt to enable advances in language models. It is not the best - it will not be the last. It is up to the community to drive its improvement, its use, and its impact. This is all but a first step into that direction.  

In this notebook, we will go through a few first examples of how we can use the dataset. The use of the dataset is by far not limited to the examples that were created therein and so I highly encourage anyone to disregard all the preprocessing, all the examples, to start from scratch, to understand deeply, to apply your best learnings, and share your results back into the open-source domain so that others may benefit and build on your shoulders.  

But now let us turn to the content of this introductory notebook and what we will do.  

1. We will explore the dataset, perform some simple analytics, and turn it into a format that is easily used for downstream machine learning tasks.
2. We will preprocess the dataset and normalize it so its content can be readily ingested into LLMs
3. We will train a continuous bag of words model (CBOW) - a word level embedding model to exemplify getting domain specific word level embeddings.
4. Going from individual words we will expand to sentences and use pre-trained models powered by Azure OpenAI to identify relationships.
5. We will then attempt to build our own sentence level embedding model and overcome the cold-start problem using OpenAIs GPT4 and a small transformer model.

None of the examples here are fleshed out, no hyper-parameters tuned, no sophisticated methods applied. None of this is the end of the road, but really the start.  

And remember it is not just about the models, but the data is where the gold may lie, how can you improve it? Make it more accesible? Make it more useful?  

The data is here - what will you build?  

## Dataset

The dataset is provided courtesy of FORCE, FORCE Sponsors, and the National Petroleum Direktorate of Norway. 
Here we process the dataset first to concatenate into one format that follows the standard used for very large datasets such as RedPyjama or ThePile.
Number of tokens (scrubbed - cl100k_base): 755829106

Maybe Jesse can contribute this part

## Preprocessing

Examples for preprocessing the dataset are provided in the `scripts` folder.

- `scripts/make_dataset.py` creates one big jsonl file from the raw dataset with metadata stored per entry.
- `scripts/make_embedding_docs.py` creates a chunked version of the dataset that can be used for computing embeddings and subsequently retrieval augmented generation. 
- `scripts/estimate_embedding_cost.py` estimates the total cost of producing embeddings from the chunked embeddings dataset.

## Installation

To install the required dependencies please create a virtual environment and install the dependencies from the requirements file:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

We have two .env files, one for shared variables that can be distributed openly, and one for the API key which will be distributed as part of the hackathon. 
`.env.shared` contains the variables we will use in the notebooks and scripts. 
`.env.secret` contains the API key for the Azure OpenAI API.

## Starter Notebook

A starter notebook can be found in the `notebooks` folder. It contains examples for loading the dataset, preprocessing, and training a CBOW model, as well as examples for using the embeddings to compute similarity and perform retrieval augmented generation.
In addition there is also an example of fine-tuning your own sentence-level embedding model. 

## Data generated from the starter notebook:

You can access a version of data that is generated from the started notebook [here]().

## License

The dataset is licensed under the <insert license here>

Code in this repository is licensed under the MIT license.

