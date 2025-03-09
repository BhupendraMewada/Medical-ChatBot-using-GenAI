from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from src.prompt import *
import os
import logging

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Initialize Embeddings
embeddings = download_hugging_face_embeddings()

# Define Pinecone Index
index_name = "medbot"

# Embed each chunk and upsert into Pinecone
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Hugging Face Inference Client
client = InferenceClient(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=HUGGINGFACE_API_KEY)

# Function to Query Model
def query_hf(prompt):
    response = client.text_generation(prompt, max_new_tokens=256, temperature=0.7)
    return response

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the RAG pipeline
question_answer_chain = create_stuff_documents_chain(query_hf, prompt)
rag_chain = RunnablePassthrough() | retriever | question_answer_chain

# Test function to check if the model is working
def test_model(prompt_text):
    try:
        logger.info(f"Testing model with prompt: {prompt_text}")

        # Pass the input to the RAG pipeline
        response = rag_chain.invoke({"input": prompt_text})
        logger.info(f"Model Response: {response}")

        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

# Main function to run the test
if __name__ == "__main__":
    # Test prompt
    test_prompt = "What is fever?"

    # Run the test
    response = test_model(test_prompt)

    if response:
        print("Model is working correctly!")
        print(f"Response: {response}")
    else:
        print("Model is not working. Check the logs for errors.")