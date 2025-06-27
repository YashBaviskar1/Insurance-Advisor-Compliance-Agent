from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Groq model name to use
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

def load_llm(groq_model_name):
    """
    Loads the Groq chat model using the LangChain integration.
    """
    llm = ChatGroq(
        temperature=0.5,
        groq_api_key=GROQ_API_KEY,
        model_name=groq_model_name,
    )
    return llm

# Custom prompt
CUSTOM_PROMPT_TEMPLATE = """
You are an Insurance QA agent whose job is to use the Context to answer to user queries 
Keep your responses relevant to the question asked and the context of the insurance policy provided
Context:
{context}

Question:
{input}

Be extensive and accurate.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "input"]
    )
    return prompt

# Load vector store
DB_PATH = "vectorstore/db_faiss_user_data"
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# Set up retriever
retriever = db.as_retriever(search_kwargs={'k': 3})

# Load LLM
llm = load_llm(GROQ_MODEL_NAME)

# Create prompt template
prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

# Create QA document chain
qa_document_chain = create_stuff_documents_chain(llm, prompt_template)

# Create RetrievalQA chain
qa_chain = create_retrieval_chain(retriever, qa_document_chain)

def get_response(user_query):
    """
    Gets a response from the QA system given a user query.
    """
    response = qa_chain.invoke({"input": user_query})
    return response['answer']
