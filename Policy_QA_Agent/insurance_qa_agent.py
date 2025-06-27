import os

# LangChain imports for loading PDFs and splitting
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---------------------------
# CONFIG
# ---------------------------
CONTEXT_PATH = "Policy_QA_Agent/uploads"
DB_PATH = "vectorstore/db_faiss_user_data"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Load environment variables

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY is None:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# ---------------------------
# Embeddings model
# ---------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# ---------------------------
# 1️⃣ Ingest PDFs and (Re)build Vectorstore
# ---------------------------
def rebuild_vectorstore_from_pdfs(upload_dir=CONTEXT_PATH, save_path=DB_PATH):
    """
    Loads PDFs from the given folder, splits them, embeds them,
    and saves the FAISS vectorstore to disk.
    """
    print(f"Loading PDFs from {upload_dir}...")
    loader = DirectoryLoader(upload_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    embedding_model = get_embedding_model()
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(save_path)
    print(f"Vectorstore saved at {save_path}.")

# ---------------------------
# 2️⃣ Load Vectorstore for Retrieval
# ---------------------------
def load_vectorstore(db_path=DB_PATH):
    embedding_model = get_embedding_model()
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# ---------------------------
# 3️⃣ Load Groq LLM
# ---------------------------
def load_llm():
    return ChatGroq(
        temperature=0.5,
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME
    )

# ---------------------------
# 4️⃣ Custom Prompt Template
# ---------------------------
CUSTOM_PROMPT_TEMPLATE = """
You are an Insurance QA agent whose job is to use the Context to answer user queries. 
Keep responses relevant to the question and the insurance policy context provided.

Context:
{context}

Question:
{input}

Be extensive and accurate.
"""

def get_prompt_template():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "input"]
    )

# ---------------------------
# 5️⃣ QA Chain Setup
# ---------------------------
def get_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={'k': 3})
    llm = load_llm()
    prompt_template = get_prompt_template()
    qa_document_chain = create_stuff_documents_chain(llm, prompt_template)
    qa_chain = create_retrieval_chain(retriever, qa_document_chain)
    return qa_chain

# ---------------------------
# 6️⃣ High-Level Functions
# ---------------------------
def rebuild_embeddings_from_upload():
    """
    Call this after new PDFs are uploaded.
    """
    rebuild_vectorstore_from_pdfs()

def get_answer(user_query):
    """
    Call this to answer questions using the existing vector store.
    """
    db = load_vectorstore()
    qa_chain = get_qa_chain(db)
    response = qa_chain.invoke({"input": user_query})
    return response['answer']

