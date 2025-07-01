# AI-powered Insurance Advisor & IRDAI Compliance Agent

Solution Video :

[![Watch the video](https://img.youtube.com/vi/kmcWhJaNvTo/0.jpg)](https://youtu.be/kmcWhJaNvTo)

An AI-powered Insurance Advisor and Compliance Agent helping users:

- Choose the right insurance
- Understand their existing policies
- Detect coverage gaps
- Stay IRDAI-compliant

[Temporary Deployed Link on GCP](http://35.207.235.123/)
(Valid till 48 hours)
Supports **CLI** and **Streamlit** interfaces.

---

## # Features

- **IRDAI Compliance Checker**  
  Analyze insurance policy PDFs for regulatory compliance against IRDAI guidelines.

- **Policy QA Agent**  
  Ask questions about your insurance documents in natural language.

- **Policy Coverage Gap Agent**  
  Analyze user profile & policy to identify coverage gaps and recommend insurance products.

- **Insurance Needs Agent**  
  Filter and recommend suitable insurance products from a CSV catalog based on user profile.

---

## Project Structure

```
.
├── insurance_qa_agent.py
├── compliance_agent.py
├── policy_coverage_gap_agent.py
├── insurance_needs_agent.py
├── vectorstore/
│ ├── db_faiss_guidelines/
│ ├── db_faiss_user_data/
│ └── db_faiss_policy_coverage_agent/
├── IRDAI_compliance_agent/
│ ├── compliance_rules.json
│ └── compliance_rules2.json
└── Policy_QA_Agent/
└── uploads/

```

---

## How it works

### Four Core Modules

---

## IRDAI Compliance Checker

> **Purpose**: Check if a user's policy document is compliant with India's IRDAI regulations.

### Data

- Official IRDAI guidelines stored as text chunks, embedded in a FAISS vector store (`vectorstore/db_faiss_guidelines`).
- Predefined rules in JSON (e.g., `IRDAI_compliance_agent/compliance_rules.json`).

### Core Steps

#### 1. Load Official Guidelines

- Stored embeddings built with `BAAI/bge-base-en-v1.5`.
- Loaded with FAISS.

#### 2. User Policy Embedding

- User uploads a PDF.
- Split into text chunks (500 characters, 50 overlap).
- Embedded with same model.
- Stored in-memory as FAISS DB.

#### 3. Similarity Check

- For each rule (category-based), do:
  - Search guideline DB for best match.
  - Search user policy DB for best match.
- Classify:
  - ✅ Compliant (score ≥ 0.75)
  - ⚠️ Partial (0.5 ≤ score < 0.75)
  - ❌ Missing (score < 0.5)

#### 4. Report

- Detailed result per rule:
  - Official guideline excerpt.
  - Closest user policy match.
  - Similarity score.
  - Compliance label.
- Overall compliance summary (percentage).

### Implementation Highlights (compliance_agent.py)

- **Vector Embeddings**:
  `HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")`
- **Vector Store**:
  - FAISS for efficient similarity search.
- **Text Splitting**:
  - RecursiveCharacterTextSplitter for chunking.
- **Configurable Rules**:
  - Rules in JSON, easy to update.

Fully modular design so new guidelines/rules can be added.

---

## Policy QA Agent

> **Purpose**: Let users ask any question about their own insurance policy PDF.

### Data

- User's policy PDFs uploaded to `Policy_QA_Agent/uploads`.
- Embedded into FAISS vector store (`vectorstore/db_faiss_user_data`).

### Core Steps

#### 1. Ingest & Embed PDFs

- DirectoryLoader for batch PDF load.
- Split into 500-character chunks.
- Embedded with `BAAI/bge-base-en-v1.5`.
- Saved as FAISS DB.

#### 2. Retrieval-Augmented QA

- Load vector store as retriever.
- Use top-k=3 matching chunks.
- Feed context into LLM.

#### 3. LLM-powered Answer Generation

- Model: Llama 4 Scout 17B

Answers are always grounded in the uploaded document context.

---

### Implementation Highlights (insurance_qa_agent.py)

- **LLM Integration**:
- Any LLM interface (ollama or API), supporting custom model.
- Environment variable for API key. (if needed)
- **Vector Store**:
- FAISS for semantic retrieval.
- **LangChain**:
- create_stuff_documents_chain
- create_retrieval_chain

Easily extensible to other insurance types or documents.

### 3Policy Coverage Gap Agent

> **Purpose**: Analyze user profile and existing policy to detect coverage gaps and recommend insurance products.

#### Data

- User profile JSON with:
- Age, income, dependents, assets, health conditions, location
- Vector store of policy text chunks

#### Core Steps

1. **Load User Profile**

- JSON input with demographics, assets, health info.

2. **Location Risk**

- Custom dictionary of Indian city-level risks.

3. **Policy Vector Store**

- FAISS DB with embedded policy chunks.

4. **LLM Analysis**

- Uses Llama 4 Scout 17B via LangChain.
- Prompt asks LLM to:
  - List rule-based & LLM-identified gaps
  - Provide risk score (0–1)
  - Prioritize gaps
  - Recommend insurance product types with example names

#### Implementation Highlights

- PromptTemplate for structured JSON output
- Groq LLM for reasoning
- Embeddings Model (BAAI/bge-base-en-v1.5)
- Modular CLI interface to load user profile and vectorstore

Outputs a JSON object with clear, actionable recommendations.

---

### Insurance Needs Agent

> **Purpose**: Recommend insurance products from a CSV catalog tailored to user profile and needs.

#### Data

- CSV file with product data:
- Age range
- Type
- Sum Assured
- Annual Premium
- Eligibility Notes

#### Core Steps

1. **Load & Clean Data**

- Parses age ranges, normalizes premiums.

2. **Filter Policies**

- Matches user age, income, needs, eligibility notes.

3. **Score Policies**

- Prefers lower premiums, higher sum assured.
- Heuristic scoring system.

4. **Recommend Top Policies**

- Returns top N recommendations sorted by score.

#### Implementation Highlights

- Uses Pandas for data cleaning/filtering
- CSV-driven – easy to extend or customize
- Independent of LLM (lightweight)

✅ Enables clear, structured recommendations for user-chosen needs and budget.

---

## Usage Modes

✅ **Streamlit UI** _(planned / demo-ready)_

- Upload PDFs.
- Choose module.
- View compliance report or chat with your policy.

  **CLI Interface**

- Run Python scripts directly:

```bash
python compliance_agent.py
python insurance_qa_agent.py
```

---

## ⚡ Tech Stack

- [LangChain](https://python.langchain.com/)
- [Hugging Face Embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [FAISS](https://faiss.ai/)
- Locally Hoted mistral-7B using ollama [for production use if needed] using LangChain
- [Groq LLM](https://groq.com/) [if hardware constraints] via LangChain
- Streamlit (for UI)
- Python 3.10+

---

## Setup

Clone the repo

```bash
git clone https://github.com/YashBaviskar1/Insurance-Advisor-Compliance-Agent.git
cd Insurance-Advisor-Compliance-Agent
```

Install dependencies

```bash
pip install -r requirements.txt
```

Set up environment variables [For quick testing, can use locally hoted ollama too]

```bash
LLM_API_KEY=your_LLM_API_key
```

Preload Vectorstore

- (Optional) Run scripts to build local FAISS DB from PDFs.

---

## Example Commands

✅ Build QA Vectorstore

```
python -c "import insurance_qa_agent; insurance_qa_agent.rebuild_embeddings_from_upload()"
```

✅ Ask a Question

```python
python -c "import insurance_qa_agent; print(insurance_qa_agent.get_answer('What is my sum insured?'))"
```

✅ Run Compliance Check

```python
from compliance_agent import *
user_db = embed_user_policy('my_policy.pdf')
guideline_db = load_guideline_db()
results, summary = run_compliance_check('Health Insurance', user_db, guideline_db)
print(summary)
```

✅ Run Coverage Gap Agent

```bash
python policy_coverage_gap_agent.py --profile user_profile.json --vectorstore vectorstore/db_faiss_policy_coverage_agent
```

✅ Use Insurance Needs Agent (example)

```python
from insurance_needs_agent import InsuranceRecommender
recommender = InsuranceRecommender("products.csv")
result = recommender.recommend({
    "age": 35,
    "income": 800000,
    "needs": ["health", "term life"],
    "max_premium": 15000
})
print(result)
```

---

## ❤️ Why this matters

Insurance in India can be opaque. Staying IRDAI-compliant protects consumers and insurers alike. This project is designed to make:

- Reading your policy easier.
- Finding gaps simpler.
- Staying compliant automatic.

---

## 📌 Future Plans

- Add more IRDAI categories.
- Integrate external guidelines sources.
- Multi-language support for rural/beema yojna [setup built]

--- -->

## Acknowledgements

- IRDAI official documentation.
- LangChain, HuggingFace, FAISS.
- Any accessible LLM APIs for quick interfaces

---

## 🏁 Hackathon Notes

This repo is designed for **DSW HACKATHON**:

- Focus on clarity and modularity.
- Easy to extend / test.
- Supports local development or demo.

---

Note : MVP testing, can be scaled to prdoction after optimization with guidelines, the python version 3.12 and Langchain dependies can cause issues
can email me at yashbav24@gmail.com

#### Built with ❤️ by Yash Baviskar
