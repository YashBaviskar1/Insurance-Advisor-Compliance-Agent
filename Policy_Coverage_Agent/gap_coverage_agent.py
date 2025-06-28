import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def save_raw_output(content: str):
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(content)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2
    )

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

TEMPLATE = """
You are an expert insurance advisor and risk analyst. Given the following user profile and policy context, identify coverage gaps and recommend suitable insurance products.

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

def load_policy_store(persist_dir: str = "vectorstore/db_faiss_policy_coverage_agent") -> FAISS:
    embedding_fn = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    store = FAISS.load_local(persist_dir, embedding_fn, allow_dangerous_deserialization=True)
    return store

def get_gap_recommendations(user_profile: dict, policy_store: FAISS) -> None:
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

    save_raw_output(response.content)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect insurance coverage gaps & get recommendations.")
    parser.add_argument("--profile", type=str, required=True, help="Path to user profile JSON file.")
    parser.add_argument("--vectorstore", type=str, default="vectorstore/db_faiss_policy_coverage_agent", help="Path to persisted FAISS policy vector store.")
    args = parser.parse_args()

    with open(args.profile) as f:
        user_profile = json.load(f)

    store = load_policy_store(args.vectorstore)
    get_gap_recommendations(user_profile, store)
