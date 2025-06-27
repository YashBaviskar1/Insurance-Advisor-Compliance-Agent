from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CONTEXT_PATH = "IRDAI_compliance_agent\guidelines_documentation"

def load_pdfs(data) :
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdfs(CONTEXT_PATH)

print(len(documents))

def create_chunks(data) :
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

text_chunks = create_chunks(documents)
print(len(text_chunks))

def get_embedding_model() :
    embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-base-en-v1.5")
    return embedding_model

embedding_model = get_embedding_model()


DB_PATH = "vectorstore/db_faiss_guidelines"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_PATH)

