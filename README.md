# 🩺 Medical Chatbot using Generative AI  

🚀 This project is an AI-powered **Medical Chatbot** that leverages **Retrieval-Augmented Generation (RAG)** to provide **context-aware** and **accurate** medical responses. It integrates **TinyLlama-1.1B-Chat** as the language model and **Pinecone** as the vector store for efficient retrieval.  

## 📌 Features  
- ✅ **AI-Powered Medical Assistance** – Provides informative responses to medical queries.  
- ✅ **Retrieval-Augmented Generation (RAG)** – Fetches relevant medical information before generating responses.  
- ✅ **TinyLlama-1.1B-Chat LLM** – Lightweight and efficient for conversational AI.  
- ✅ **Pinecone Vector Search** – Fast and scalable similarity search for embeddings.  
- ✅ **Hugging Face Model API** – Easily integrates models for inference.  
- ✅ **Flask Web Interface** – Simple and interactive chatbot interface.  

## 🛠 Tech Stack  
| **Component**     | **Technology**                                      |
|------------------|--------------------------------------------------|
| **Backend**      | Flask, LangChain, Hugging Face API               |
| **LLM Model**    | TinyLlama-1.1B-Chat (via Hugging Face)           |
| **Vector DB**    | Pinecone                                         |
| **Embeddings**   | sentence-transformers/all-MiniLM-L6-v2           |
| **Frontend**     | HTML, CSS, JavaScript                            |

## 📖 How It Works  
The chatbot follows a **Retrieval-Augmented Generation (RAG)** approach:  
1. **User Input** – The chatbot receives a medical query.  
2. **Vector Search** – Converts the query into an embedding and retrieves relevant documents from **Pinecone**.  
3. **LLM Processing** – The **TinyLlama-1.1B-Chat** model generates an AI-powered response using the retrieved information.  
4. **Response Generation** – The chatbot returns a well-informed medical response.  

---
