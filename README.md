# 🛡️ AI-powered Insurance Advisor & IRDAI Compliance Agent

An AI-powered Insurance Advisor and Compliance Agent helping users:

✅ Choose the right insurance  
✅ Understand their existing policies  
✅ Detect coverage gaps  
✅ Stay IRDAI-compliant

Supports **CLI** and **Streamlit** interfaces.

---

## 🚀 Features

- **IRDAI Compliance Checker**  
  Analyze insurance policy PDFs for regulatory compliance against IRDAI guidelines.

- **Policy QA Agent**  
  Ask questions about your insurance documents in natural language.

---

## 📦 Project Structure

```
.
├── insurance_qa_agent.py
├── compliance_agent.py
├── vectorstore/
├── IRDAI_compliance_agent/
│   ├── compliance_rules.json
│   └── compliance_rules2.json
└── Policy_QA_Agent/
    └── uploads/
```

---

## 🧠 How it works

### 🗂️ Two Core Modules

---

## 1️⃣ IRDAI Compliance Checker

> **Purpose**: Check if a user's policy document is compliant with India's IRDAI regulations.

### 🏛️ Data

- Official IRDAI guidelines stored as text chunks, embedded in a FAISS vector store (`vectorstore/db_faiss_guidelines`).
- Predefined rules in JSON (e.g., `IRDAI_compliance_agent/compliance_rules.json`).

### ⚙️ Core Steps

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

### 🧩 Implementation Highlights (compliance_agent.py)

- **Vector Embeddings**:
  `HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")`
- **Vector Store**:
  - FAISS for efficient similarity search.
- **Text Splitting**:
  - RecursiveCharacterTextSplitter for chunking.
- **Configurable Rules**:
  - Rules in JSON, easy to update.

✅ Fully modular design so new guidelines/rules can be added.

---

## 2️⃣ Policy QA Agent

> **Purpose**: Let users ask any question about their own insurance policy PDF.

### 📚 Data

- User's policy PDFs uploaded to `Policy_QA_Agent/uploads`.
- Embedded into FAISS vector store (`vectorstore/db_faiss_user_data`).

### ⚙️ Core Steps

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

- Model: Groq Llama 4 Scout 17B
  - Accessed via `langchain_groq.ChatGroq`
- Custom prompt template to ensure answers are insurance-specific:
  ```
  You are an Insurance QA agent whose job is to use the Context to answer user queries...
  ```

✅ Answers are always grounded in the uploaded document context.

---

### 🧩 Implementation Highlights (insurance_qa_agent.py)

- **LLM Integration**:
  - Groq API, supporting custom model.
  - Environment variable for API key.
- **Vector Store**:
  - FAISS for semantic retrieval.
- **LangChain**:
  - create_stuff_documents_chain
  - create_retrieval_chain

✅ Easily extensible to other insurance types or documents.

---

## 💻 Usage Modes

✅ **Streamlit UI** _(planned / demo-ready)_

- Upload PDFs.
- Choose module.
- View compliance report or chat with your policy.

✅ **CLI Interface**

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
- [Groq LLM](https://groq.com/) via LangChain
- Streamlit (for UI)
- Python 3.10+

---

## 🛠️ Setup

1️⃣ Clone the repo

```bash
git clone https://github.com/YashBaviskar1/Insurance-Advisor-Compliance-Agent.git
cd Insurance-Advisor-Compliance-Agent
```

2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

3️⃣ Set up environment variables

```bash
LLM_API_KEY=your_LLM_API_key
```

4️⃣ Preload Vectorstore

- (Optional) Run scripts to build local FAISS DB from PDFs.

---

## 🎯 Example Commands

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

---

## ❤️ Why this matters

Insurance in India can be opaque. Staying IRDAI-compliant protects consumers and insurers alike. This project is designed to make:

- Reading your policy easier.
- Finding gaps simpler.
- Staying compliant automatic.

---

<!-- ## 📌 Future Plans

- Add more IRDAI categories.
- Integrate external guidelines sources.
- Streamlit front-end with file upload and interactive reports.
- Multi-language support.

--- -->

## 🙏 Acknowledgements

- IRDAI official documentation.
- LangChain, HuggingFace, FAISS.
- Any accessible LLM APIs.

---

## 🏁 Hackathon Notes

This repo is designed for **DSW HACKATHON**:

- Focus on clarity and modularity.
- Easy to extend / test.
- Supports local development or demo.

---

#### Built with ❤️ by Yash Baviskar
