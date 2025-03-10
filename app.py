from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from huggingface_hub import InferenceClient
import os
import logging

app = Flask(__name__)
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
    # Convert ChatPromptValue to a string
    if hasattr(prompt, "to_string"):
        prompt_str = prompt.to_string()
    else:
        prompt_str = str(prompt)
    
    # Send the prompt to the Hugging Face Inference API
    response = client.text_generation(prompt_str, max_new_tokens=256, temperature=0.7)
    return response

# Wrap the query_hf function in a Runnable
query_runnable = RunnableLambda(query_hf)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(query_runnable, prompt)

# Create the RAG pipeline
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        # Get user input from the form
        msg = request.form.get("msg", "").strip()
        logger.info(f"User Input: {msg}")

        # Validate input
        if not msg:
            return jsonify({"error": "Invalid input. Please provide a valid message."}), 400

        # Pass the input to the RAG pipeline
        response = rag_chain.invoke({"input": msg})
        logger.info(f"Model Response: {response['answer']}")

        # Return the response as JSON
        return jsonify({"response": response["answer"]})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == "__main__":
    app.run()
