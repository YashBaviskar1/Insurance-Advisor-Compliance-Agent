import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# ---------------------------
# 1️⃣ Load Embeddings & Vector Store
# ---------------------------

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vector_store = FAISS.load_local(
    "vectorstore/db_faiss_guidelines",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 8})

# ---------------------------
# 2️⃣ Define Prompt
# ---------------------------

RULE_PROMPT_TEMPLATE = """You are an insurance regulatory compliance expert.
Extract specific compliance requirements from the following IRDAI guideline excerpts.

Guidelines:
{context}

Answer:"""

RULE_PROMPT = PromptTemplate(
    template=RULE_PROMPT_TEMPLATE,
    input_variables=["context"]
)

# ---------------------------
# 3️⃣ Load LLM
# ---------------------------

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    model_kwargs={
        "token": HF_TOKEN,
        "max_length": 1024
    }
)

# ---------------------------
# 4️⃣ Create RetrievalQA Chain
# ---------------------------

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": RULE_PROMPT},
    input_key="query",
    output_key="result"
)

# ---------------------------
# 5️⃣ Compliance Rule Generation Function (just text)
# ---------------------------

def generate_compliance_text(query=None):
    if not query:
        query = (
            "List key compliance requirements for insurance policies with "
            "numerical limits, deadlines, and specific obligations"
        )

    try:
        result = qa_chain.invoke({"query": query})
        return result["result"]
    except Exception as e:
        print(f"Error generating compliance rules: {str(e)}")
        return ""

# ---------------------------
# 6️⃣ Example usage
# ---------------------------

if __name__ == "__main__":
    test_query = """Extract compliance rules about:
    1. Fire insurance sum insured calculations
    2. Motor insurance premium discounts
    3. Claim settlement timelines"""
    
    compliance_text = generate_compliance_text(query=test_query)

    # Write to txt file
    output_file = "IRDAI_compliance_agent/compliance_rules_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(compliance_text)

    print(f"Compliance rules saved to {output_file}")
