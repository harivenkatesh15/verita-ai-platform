import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

# 1. Specialized integration imports
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# 2. Modern RAG Chain imports (requires langchain-classic)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 3. Import helper functions
from src.helper import download_embeddings
from src.prompt import *

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set environmental variables for LangChain integrations
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize components
embeddings = download_embeddings()
index_name = "medical-chatbot"

# Load existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize Groq LLM
chatModel = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    groq_api_key=GROQ_API_KEY
)

# Prompt Definition
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Build Chains
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)