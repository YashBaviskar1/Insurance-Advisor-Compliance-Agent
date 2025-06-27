import os
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === CONFIG ===
GUIDELINES_DB_PATH = "vectorstore/db_faiss_guidelines"
RAW_PDF_PATH = "data/guidelines_documentation"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SIMILARITY_THRESHOLD_COMPLIANT = 0.75
SIMILARITY_THRESHOLD_PARTIAL = 0.5
COMPLIANCE_RULE_FILES = [
    "IRDAI_compliance_agent/compliance_rules.json",
    "IRDAI_compliance_agent/compliance_rules2.json"
]
COMMON_CATEGORIES = ["Claims", "Consumer Rights"]

# === EMBEDDING MODEL ===
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# === Load compliance rules once ===
with open(COMPLIANCE_RULE_FILES[0], 'r', encoding='utf-8') as f:
    rules1 = json.load(f)
with open(COMPLIANCE_RULE_FILES[1], 'r', encoding='utf-8') as f:
    rules2 = json.load(f)

ALL_RULES = rules1 + rules2

def get_available_categories():
    return sorted(set(
        rule["category"]
        for rule in ALL_RULES
        if rule["category"] not in COMMON_CATEGORIES
    ))

# === Runtime FAISS builder ===
def build_guidelines_vectorstore():
    print("[INFO] Building FAISS guidelines store from PDFs...")
    loader = DirectoryLoader(RAW_PDF_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Created {len(chunks)} text chunks.")

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(GUIDELINES_DB_PATH)
    print(f"[INFO] FAISS store saved to {GUIDELINES_DB_PATH}")

def load_guideline_db():
    """Load or build the IRDAI guidelines vectorstore."""
    if not os.path.exists(GUIDELINES_DB_PATH):
        build_guidelines_vectorstore()

    return FAISS.load_local(
        GUIDELINES_DB_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

def embed_user_policy(pdf_path):
    """Load and embed the user-uploaded policy PDF."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    user_chunks = splitter.split_documents(documents)
    user_texts = [chunk.page_content for chunk in user_chunks]

    user_db = FAISS.from_texts(user_texts, embedding_model)
    return user_db

def classify_similarity(score):
    if score >= SIMILARITY_THRESHOLD_COMPLIANT:
        return "✅ Compliant"
    elif score >= SIMILARITY_THRESHOLD_PARTIAL:
        return "⚠️ Partial"
    else:
        return "❌ Missing"

def run_compliance_check(category, user_db, guideline_db):
    """Run compliance checks and return results + summary."""
    selected_rules = [
        rule for rule in ALL_RULES
        if rule["category"] == category or rule["category"] in COMMON_CATEGORIES
    ]

    results = []
    for rule in selected_rules:
        query = rule["description"]

        guideline_result = guideline_db.similarity_search_with_score(query, k=1)
        guideline_text, guideline_score = guideline_result[0][0].page_content, guideline_result[0][1]

        user_result = user_db.similarity_search_with_score(query, k=1)
        user_text, user_score = user_result[0][0].page_content, user_result[0][1]

        compliance_status = classify_similarity(user_score)

        results.append({
            "id": rule["id"],
            "category": rule["category"],
            "description": rule["description"],
            "status": compliance_status,
            "official_guideline": guideline_text,
            "user_policy_text": user_text,
            "similarity_score": user_score
        })

    num_total = len(results)
    num_compliant = sum(1 for r in results if r["status"] == "✅ Compliant")
    compliance_score = (num_compliant / num_total) * 100 if num_total > 0 else 0.0

    summary = {
        "total_checks": num_total,
        "compliant": num_compliant,
        "compliance_score": compliance_score
    }

    return results, summary
