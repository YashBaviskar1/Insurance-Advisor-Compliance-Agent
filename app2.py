# app2.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize session state
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Load vector store
@st.cache_resource
def load_vector_store():
    DB_PATH = "vectorstore/db_faiss_beema"
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# Initialize QA chain
def init_qa_chain(language="hindi"):
    # Load LLM with language-specific model
    groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"  # Supports Hindi
    
    llm = ChatGroq(
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
        model_name=groq_model
    )
    
    # Language-specific instructions
    lang_instruction = {
        "hindi": "केवल हिंदी में उत्तर दें। सभी प्रतिक्रियाएँ हिंदी में होनी चाहिए।",
        "english": "Respond only in English. All answers must be in English."
    }
    
    # Custom prompt with user profile and language
    PROMPT_TEMPLATE = f"""
    आप एक बीमा सलाहकार हैं। उपयोगकर्ता के प्रश्न का संदर्भ और उपयोगकर्ता प्रोफ़ाइल का उपयोग करके उत्तर दें।
    
    {lang_instruction[language]}
    
    उपयोगकर्ता प्रोफ़ाइल:
    {st.session_state.user_profile}
    
    संदर्भ:
    {{context}}
    
    प्रश्न:
    {{input}}
    
    विस्तृत और सटीक उत्तर दें। केवल संदर्भ में दी गई जानकारी का उपयोग करें।
    """
    
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "input"]
    )
    
    # Create QA chain
    db = load_vector_store()
    retriever = db.as_retriever(search_kwargs={'k': 3})
    qa_document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_document_chain)

# User profile form
def user_profile_form():
    with st.form("user_profile_form"):
        st.subheader("👤 अपना विवरण दर्ज करें")
        
        col1, col2 = st.columns(2)
        age = col1.number_input("उम्र", min_value=18, max_value=100, value=35)
        income = col2.selectbox("वार्षिक आय", 
                              ["< ₹5 लाख", "₹5-10 लाख", "₹10-20 लाख", "> ₹20 लाख"])
        
        dependents = st.multiselect("आश्रित (परिवार के सदस्य)", 
                                  ["जीवनसाथी", "बच्चे", "माता-पिता", "अन्य"])
        
        assets = st.multiselect("संपत्ति", 
                              ["घर", "कार", "जमीन", "व्यवसाय"])
        
        health = st.selectbox("स्वास्थ्य स्थिति", 
                            ["उत्कृष्ट", "अच्छा", "उच्च रक्तचाप", "मधुमेह", "हृदय रोग"])
        
        location = st.selectbox("स्थान", 
                              ["मुंबई", "दिल्ली", "चेन्नई", "बैंगलोर", "कोलकाता", "हैदराबाद"])
        
        language = st.radio(
    "भाषा", 
    ["hindi", "english"], 
    format_func=lambda x: "हिंदी" if x=="hindi" else "अंग्रेजी"
)
        
        if st.form_submit_button("प्रोफ़ाइल सहेजें"):
            st.session_state.user_profile = {
                "age": age,
                "income": income,
                "dependents": dependents,
                "assets": assets,
                "health": health,
                "location": location,
                "language": language.lower()
            }
            st.success("प्रोफ़ाइल सफलतापूर्वक सहेजी गई!")
            st.session_state.qa_chain = init_qa_chain(language.lower())

# Main QA interface
def qa_interface():
    st.subheader("❓ अपने बीमा के बारे में पूछें")
    
    # Initialize QA chain if not done
    if st.session_state.qa_chain is None:
        st.session_state.qa_chain = init_qa_chain(
            st.session_state.user_profile.get("language", "hindi")
        )
    
    # Display user profile
    if st.session_state.user_profile:
        with st.expander("आपकी प्रोफ़ाइल"):
            st.json(st.session_state.user_profile)
    
    # Question input
    question = st.text_area("अपना प्रश्न दर्ज करें:", 
                          placeholder="मेरे स्वास्थ्य बीमा में अस्पताल में भर्ती होने की अवधि क्या है?")
    
    if st.button("उत्तर प्राप्त करें") and question:
        with st.spinner("आपके प्रश्न का विश्लेषण किया जा रहा है..."):
            try:
                response = st.session_state.qa_chain.invoke({"input": question})
                st.subheader("उत्तर:")
                st.markdown(f"<div style='background-color:#f0f2f6; padding:20px; border-radius:10px;'>{response['answer']}</div>", 
                           unsafe_allow_html=True)
                
                # Show context sources
                with st.expander("संदर्भ स्रोत देखें"):
                    for i, doc in enumerate(response["context"]):
                        st.caption(f"स्रोत {i+1}:")
                        st.text(doc.page_content[:500] + "...")
            except Exception as e:
                st.error(f"त्रुटि: {str(e)}")

# Main app
def main():
    st.set_page_config(page_title="भारत बीमा सहायक", page_icon="🛡️")
    st.title("🛡️ भारत बीमा सहायक")
    st.caption("AI-पावर्ड बीमा सलाहकार जो भारतीय सरकारी योजनाओं की व्याख्या करता है")
    
    # Initialize vector store
    try:
        load_vector_store()
    except:
        st.error("वेक्टर स्टोर लोड करने में त्रुटि। कृपया सुनिश्चित करें कि आपने पीडीएफ़ इनजेस्ट किया है")
        return
    
    # Show user profile form if not completed
    if not st.session_state.user_profile:
        user_profile_form()
    else:
        if st.button("प्रोफ़ाइल संपादित करें"):
            st.session_state.user_profile = {}
            st.rerun()
        qa_interface()

if __name__ == "__main__":
    main()