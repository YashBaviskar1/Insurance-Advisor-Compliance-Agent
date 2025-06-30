import os
import shutil
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ========== CONFIG ==========
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
VECTORSTORE_PATH = "vectorstore/db_faiss_policy_coverage_agent"

# ========== RISK LOOKUP ==========
LOCATION_RISKS = {
    "Mumbai": "flood-prone coastal area with seismic activity (Zone III)",
    "Delhi": "high air pollution, urban flooding, and seismic risk (Zone IV)",
    "Chennai": "cyclone-prone coastal flood zone in seismic Zone III",
    "Kerala": "flood and landslide-prone high rainfall, seismic Zone III",
    "Kolkata": "cyclone and flood risk in low-elevation coastal area, seismic Zone III",
    "Bangalore": "low natural disaster risk but high urban stress and occasional flooding",
    "Hyderabad": "flash flooding from heavy monsoon rains due to poor drainage",
    "Pune": "urban flooding risks from monsoon rains and hill‑slope terrain",
    "Ahmedabad": "acute heatwaves, drought-prone semi‑arid region",
    "Guwahati": "very high vulnerability to earthquakes, floods, and storms",
    "Lucknow": "flood risk from monsoon and moderate seismic activity (Zone III)",
    "Patna": "recurrent flooding from Himalayan river systems"
}

def get_location_risk(location: str) -> str:
    return LOCATION_RISKS.get(location.strip(), f"general {location} area with typical risks")

# ========== EMBEDDING MODEL LOADER =========
import streamlit as st
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}  # Use GPU if available: 'cuda'
    )


# ========== INGESTION FUNCTIONS ==========
def clear_vectorstore(path: str = VECTORSTORE_PATH):
    """Delete the existing FAISS DB directory if it exists."""
    if os.path.exists(path):
        shutil.rmtree(path)

def ingest_policies_from_directory(pdf_directory: str, persist_path: str = VECTORSTORE_PATH):
    # Only clear store if it exists and we want to rebuild completely
    if not os.path.exists(persist_path):
        os.makedirs(persist_path)
    
    embeddings = get_embedding_model()
    
    # Try to load existing DB if available
    try:
        db = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    except:
        db = None
    
    # Load PDFs
    loader = DirectoryLoader(pdf_directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(documents)
        
        if db:
            db.add_documents(text_chunks)
        else:
            db = FAISS.from_documents(text_chunks, embeddings)
        
        db.save_local(persist_path)
        return len(text_chunks)
    return 0
# ========== POLICY STORE LOADER ==========
def load_policy_store(persist_path: str = VECTORSTORE_PATH) -> FAISS:
    embeddings = get_embedding_model()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)

# ========== LLM & PROMPT ==========
TEMPLATE = """
You are an expert insurance advisor and risk analyst. Given the following user profile and policy context, identify coverage gaps and recommend suitable insurance products.
Do not say anything, just give JSON output 
User Profile:
- Age: {age}
- Dependents: {dependents}
- Income: {income}
- Assets: {assets}
- Health Conditions: {health_conditions}
- Location Risk: {location_risk}

Policy Context:
{policy_snippets}

Tasks:
1. List rule-based and LLM-identified coverage gaps in JSON {{"rule_based_gaps": [...], "llm_gaps": [...]}}.
2. Provide a risk score between 0 and 1.
3. Prioritize gaps and output array "priority_gaps".
4. Recommend insurance product types and example names for each gap.

Output a single JSON object:
{{
  "rule_based_gaps": [...],
  "llm_gaps": [...],
  "risk_score": 0.0,
  "priority_gaps": [...],
  "recommendations": [{{"type": "term life", "example": "Max Life Smart Term Plan", "justification": "..."}}, ...]
}}
"""

prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=[
        "age", "dependents", "income", "assets",
        "health_conditions", "location_risk", "policy_snippets"
    ]
)

def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2
    )
import re
import json
import ast

def extract_json(text):
    # Try extracting a JSON code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Or fallback to whole text
    return text.strip()

def safe_parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            raise ValueError("LLM output is not valid JSON or Python dict")

def get_gap_recommendations(user_profile: dict, policy_store: FAISS) -> dict:
    """
    Given user profile dict and loaded FAISS store, return the parsed JSON recommendation.
    Raises ValueError if LLM output is invalid JSON.
    """
    location_risk = get_location_risk(user_profile.get("location", ""))
    docs = policy_store.similarity_search(query="coverage terms", k=5)
    snippets = "\n---\n".join([doc.page_content for doc in docs])

    llm = load_llm()
    chain = prompt | llm
    response = chain.invoke({
        "age": user_profile.get("age"),
        "dependents": user_profile.get("dependents"),
        "income": user_profile.get("income"),
        "assets": user_profile.get("assets"),
        "health_conditions": user_profile.get("health_conditions"),
        "location_risk": location_risk,
        "policy_snippets": snippets
    })

    # ✅ FIX: Ensure we extract text
    if hasattr(response, "content"):
        response = response.content
    elif hasattr(response, "text"):
        response = response.text
    else:
        response = str(response)

    # Try parsing to JSON
    try:
        clean_text = extract_json(response)
        return safe_parse_json(clean_text)
    except json.JSONDecodeError:
        # Optionally: Save raw output for inspection
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(response)
        raise ValueError("LLM output is not valid JSON. Saved to output.txt.")
