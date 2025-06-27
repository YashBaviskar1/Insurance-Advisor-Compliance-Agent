import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

GUIDELINES_DB_PATH = "vectorstore/db_faiss_guidelines"
USER_POLICY_PATH = r"IRDAI_compliance_agent\uploads\standard-fire-special-perils-policy-wordings.pdf"
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

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

print("[INFO] Loading IRDAI guidelines vectorstore...")
guideline_db = FAISS.load_local(GUIDELINES_DB_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)

def load_and_embed_user_policy(pdf_path):
    print(f"[INFO] Loading user policy PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("[INFO] Splitting user policy into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    user_chunks = splitter.split_documents(documents)
    user_texts = [chunk.page_content for chunk in user_chunks]

    print("[INFO] Creating temporary user policy FAISS store...")
    user_db = FAISS.from_texts(user_texts, embedding_model)
    return user_db

user_policy_db = load_and_embed_user_policy(USER_POLICY_PATH)

print("[INFO] Loading compliance rules from JSON...")
all_rules = []
for rule_file in COMPLIANCE_RULE_FILES:
    with open(rule_file, 'r', encoding='utf-8') as f:
        rules = json.load(f)
        all_rules.extend(rules)

print(f"[INFO] Loaded {len(all_rules)} compliance checks.")


# ============ NEW PART: Ask user to choose category ============

# Build list of available categories excluding COMMON
all_categories = sorted(set(
    rule["category"] for rule in all_rules if rule["category"] not in COMMON_CATEGORIES
))

print("\n======= AVAILABLE POLICY CATEGORIES =======")
for idx, cat in enumerate(all_categories, 1):
    print(f"{idx}. {cat}")

while True:
    try:
        choice = int(input("\nSelect the category number for your policy: ")) - 1
        if 0 <= choice < len(all_categories):
            selected_category = all_categories[choice]
            break
        else:
            print("Invalid choice. Try again.")
    except ValueError:
        print("Invalid input. Enter a number.")

print(f"\n[INFO] Selected category: {selected_category}")
print(f"[INFO] Common categories always included: {COMMON_CATEGORIES}")

# Filter rules for selected category + common
selected_rules = [
    rule for rule in all_rules 
    if rule["category"] == selected_category or rule["category"] in COMMON_CATEGORIES
]

print(f"[INFO] Total rules to check after filtering: {len(selected_rules)}")


# ============ COMPLIANCE CHECKS ============

def classify_similarity(score):
    if score >= SIMILARITY_THRESHOLD_COMPLIANT:
        return "✅ Compliant"
    elif score >= SIMILARITY_THRESHOLD_PARTIAL:
        return "⚠️ Partial"
    else:
        return "❌ Missing"

results = []

print("\n[INFO] Running compliance checks...\n")
for rule in selected_rules:
    query = rule["description"]

    # Search official IRDAI guideline DB
    guideline_result = guideline_db.similarity_search_with_score(query, k=1)
    guideline_text, guideline_score = guideline_result[0][0].page_content, guideline_result[0][1]

    # Search user policy DB
    user_result = user_policy_db.similarity_search_with_score(query, k=1)
    user_text, user_score = user_result[0][0].page_content, user_result[0][1]

    # Classify
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
compliance_score = (num_compliant / num_total) * 100

print("\n======= COMPLIANCE CHECK RESULTS =======\n")
for r in results:
    print(f"[{r['id']}] {r['description']}")
    print(f"  → Status: {r['status']}")
    print(f"  → Similarity Score: {r['similarity_score']:.2f}")
    print(f"  → Official Guideline Snippet: {r['official_guideline'][:150]}...")
    print(f"  → User Policy Snippet: {r['user_policy_text'][:150]}...")
    print("------------------------------------------------------")

print("\n======= SUMMARY =======")
print(f"✅ Total checks: {num_total}")
print(f"✅ Compliant: {num_compliant}")
print(f"⚠️  Score: {compliance_score:.2f}%")

print("\n======= RECOMMENDATIONS =======")
for r in results:
    if r["status"] != "✅ Compliant":
        print(f"- [{r['id']}] {r['description']}")
        print(f"  Recommended Action: Add/clarify as per guideline → {r['official_guideline'][:200]}...")
        print("")

# Save to text file
with open("IRDAI_compliance_agent/test3_new.txt", "w", encoding="utf-8") as f:
    f.write("\n======= COMPLIANCE CHECK RESULTS =======\n\n")
    for r in results:
        f.write(f"[{r['id']}] {r['description']}\n")
        f.write(f"  → Status: {r['status']}\n")
        f.write(f"  → Similarity Score: {r['similarity_score']:.2f}\n")
        f.write(f"  → Official Guideline Snippet: {r['official_guideline'][:150]}...\n")
        f.write(f"  → User Policy Snippet: {r['user_policy_text'][:150]}...\n")
        f.write("------------------------------------------------------\n")

    f.write("\n======= SUMMARY =======\n")
    f.write(f"✅ Total checks: {num_total}\n")
    f.write(f"✅ Compliant: {num_compliant}\n")
    f.write(f"⚠️  Score: {compliance_score:.2f}%\n")

    f.write("\n======= RECOMMENDATIONS =======\n")
    for r in results:
        if r["status"] != "✅ Compliant":
            f.write(f"- [{r['id']}] {r['description']}\n")
            f.write(f"  Recommended Action: Add/clarify as per guideline → {r['official_guideline'][:200]}...\n\n")
