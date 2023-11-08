from openai import AzureOpenAI
from openai.types import Embedding
from dotenv import load_dotenv
import pytest
import os
import logging
# Load the shared environment variables, not secrets
load_dotenv(".env.shared")

try:
    # Check if we are on github
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    logging.info("Loaded API key from environment")
except KeyError:
    # Apparently we are local, so lets load the key
    load_dotenv(".env.secret")
    logging.info("Loaded API key from .env.secret")


@pytest.fixture
def client():
    logging.info("OPENAI_API_BASE="+os.environ["OPENAI_API_BASE"])
    openai_client = AzureOpenAI(
        api_version=os.environ["OPENAI_API_VERSION"],
        azure_endpoint=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"]
    )
    return openai_client


@pytest.mark.parametrize("model_deployment", [os.environ["GPT35Turbo_DEPLOYMENT"], 
                                              os.environ["GPT35Turbo16k_DEPLOYMENT"], 
                                              os.environ["GPT4_DEPLOYMENT"],
                                              os.environ["GPT432k_DEPLOYMENT"]])
def test_chat_completion(client, model_deployment):
    """
    Test the chat completion API
    """
    completion = client.chat.completions.create(model=model_deployment, 
                                            messages=[{"role": "user", "content": "Hello world"}])
    assert isinstance(completion.choices[0].message.content, str)


def test_completion(client):
    """
    Test the completion API
    """
    completion = client.completions.create(model=os.environ["DAVINCI002_DEPLOYMENT"],
                                           prompt="Hello world")
    assert isinstance(completion.choices[0].text, str)


def test_embedding(client):
    """
    Test the embedding API
    """
    embedding = client.embeddings.create(input=["sample text", "other sample text"], 
                                        model=os.environ["ADA002_DEPLOYMENT"])
    
    assert isinstance(embedding.data, list)
    assert isinstance(embedding.data[0], Embedding)
