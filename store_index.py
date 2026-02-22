
import os
from dotenv import load_dotenv
from src.helper import load_pdfs, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY



extracted_data = load_pdfs(data_path='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)


embeddings = download_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"


if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, # Must match the HuggingFace model dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)

print("Success! Your medical knowledge base is now live in Pinecone.")




