import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from flask_cors import CORS

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.helper import download_embeddings
# Import BOTH prompts
from src.prompt import patient_system_prompt, doctor_system_prompt

app = Flask(__name__)
CORS(app, origins=["*"])

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    groq_api_key=GROQ_API_KEY
)

# --- GUARDRAIL 1: Fast Pre-flight Check (Patients Only) ---
EMERGENCY_KEYWORDS = ["suicide", "kill myself", "end my life", "heart attack", "chest pain", "stroke", "can't breathe", "severe bleeding", "overdose"]
EDUCATIONAL_BYPASS = ["what are", "symptoms", "signs of", "define", "explain", "how to", "difference between"]

def check_emergency(msg: str) -> str:
    msg_lower = msg.lower()
    if any(keyword in msg_lower for keyword in EMERGENCY_KEYWORDS):
        is_educational = any(bypass in msg_lower for bypass in EDUCATIONAL_BYPASS)
        is_personal = "i have" in msg_lower or "my" in msg_lower or "i am" in msg_lower
        if is_educational and not is_personal:
            return None 
        return "🚨 This sounds like a medical emergency. Please call 108 immediately or go to the nearest hospital. I am an AI and cannot assist with emergencies."
    return None

# --- GUARDRAIL 2: Domain Classification Chain (Both Users) ---
guardrail_prompt = PromptTemplate.from_template(
    """You are a strict security classifier for a medical AI. 
    Is the following query related to human health, medicine, biology, wellness, diet, or symptoms?
    If it is a greeting (e.g., "hi", "hello"), allow it.
    If it is asking to write code, solve math, write a story, or discuss non-medical topics, reject it.
    Answer ONLY with 'YES' or 'NO'.
    Query: {query}"""
)
domain_guardrail_chain = guardrail_prompt | chatModel | StrOutputParser()

# --- GUARDRAIL 3: Role-Based RAG Chains ---
# Chain A: For Patients
patient_prompt = ChatPromptTemplate.from_messages([("system", patient_system_prompt), ("human", "{input}")])
patient_qa_chain = create_stuff_documents_chain(chatModel, patient_prompt)
patient_rag_chain = create_retrieval_chain(retriever, patient_qa_chain)

# Chain B: For Doctors
doctor_prompt = ChatPromptTemplate.from_messages([("system", doctor_system_prompt), ("human", "{input}")])
doctor_qa_chain = create_stuff_documents_chain(chatModel, doctor_prompt)
doctor_rag_chain = create_retrieval_chain(retriever, doctor_qa_chain)


@app.route("/")
def index():
    return render_template('web.html')

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    msg = data.get("message", "")
    role = data.get("role", "patient") # Defaults to patient if not provided
    
    if not msg:
        return jsonify({"error": "message is required"}), 400
    
    # Layer 1: Emergency Check (APPLIES TO PATIENTS ONLY)
    if role == "patient":
        emergency_response = check_emergency(msg)
        if emergency_response:
            return jsonify({"answer": emergency_response})
    
    # Layer 2: Domain Classifier (APPLIES TO BOTH - prevents coding/math jailbreaks)
    try:
        is_medical = domain_guardrail_chain.invoke({"query": msg})
        if "NO" in is_medical.upper():
            return jsonify({
                "answer": "I am a specialized medical assistant. I can only answer questions related to health, wellness, and medicine."
            })
    except Exception as e:
        print(f"Guardrail error: {e}")

    # Layer 3: Execute the correct RAG Chain
    try:
        if role == "doctor":
            response = doctor_rag_chain.invoke({"input": msg})
        else:
            response = patient_rag_chain.invoke({"input": msg})
            
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        print(f"RAG error: {e}")
        return jsonify({"answer": "An error occurred while accessing the medical database. Please try again later."}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)