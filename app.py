from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone
from langchain_core.runnables import RunnableLambda
from langchain_chains.combine_documents import create_stuff_documents_chain
from langchain_chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from huggingface_hub import InferenceClient
import os
import logging
import time

app = Flask(__name__)
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Initialize Embeddings (Load once)
embeddings = download_hugging_face_embeddings()

# Define Pinecone Index
index_name = "medbot"

# Initialize Pinecone Vector Store (Load once)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Initialize Retriever with fewer docs to speed up retrieval
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Initialize Hugging Face Inference Client (Load once)
client = InferenceClient(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=HUGGINGFACE_API_KEY)

# Function to Query Model with optimizations
def query_hf(prompt):
    try:
        if hasattr(prompt, "to_string"):
            prompt_str = prompt.to_string()
        else:
            prompt_str = str(prompt)
        
        start_time = time.time()
        response = client.text_generation(prompt_str, max_new_tokens=128, temperature=0.7)
        elapsed_time = time.time() - start_time
        logger.info(f"Response Time: {elapsed_time:.2f}s")
        return response
    except Exception as e:
        logger.error(f"Error in query_hf: {e}")
        return "An error occurred while generating the response."

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
        msg = request.form.get("msg", "").strip()
        logger.info(f"User Input: {msg}")
        
        if not msg:
            return jsonify({"error": "Invalid input. Please provide a valid message."}), 400
        
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Error: No answer generated.")
        
        logger.info(f"Model Response: {answer}")
        return jsonify({"response": answer})
    except Exception as e:
        logger.error(f"Error in /get endpoint: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == "__main__":
    app.run(threaded=True, debug=False)
