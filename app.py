import streamlit as st
from IRDAI_compliance_agent import compliance_agent
from Policy_QA_Agent import insurance_qa_agent 
import tempfile
import os

# -----------------------------------------
# Page metadata
# -----------------------------------------
st.set_page_config(
    page_title="Insurance Advisor and Compliance Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# Custom CSS
# -----------------------------------------
st.markdown("""
    <style>
        .main-title {
            color: #004d99;
            font-size: 3em;
            font-weight: bold;
            text-align: center;
        }
        .sub-title {
            color: #0066cc;
            font-size: 1.2em;
            text-align: center;
        }
        .footer {
            text-align: center;
            color: grey;
            font-size: 0.9em;
            margin-top: 50px;
        }
        .center-section {
            text-align: center;
            font-size: 1.2em;
            max-width: 800px;
            margin: 0 auto;
        }
        .section-header {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Header
# -----------------------------------------
st.markdown('<div class="main-title">🛡️ Insurance Advisor and Compliance Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An AI-powered assistant to help you choose the right insurance, understand policies, detect gaps, and stay IRDAI compliant.</div>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------------
# Sidebar navigation
# -----------------------------------------
with st.sidebar:
    st.title("🗂️ Navigation")
    app_mode = st.radio(
        "Choose a module:",
        ["IRDAI Compliance Checker", "Policy QA Agent"]
    )
    st.markdown("---")
    st.markdown("**ℹ️ About this app**")
    st.caption("This AI tool assists users in insurance advisory, policy question-answering, and IRDAI compliance checking.")

# -----------------------------------------
# Load guideline DB once (cached)
# -----------------------------------------
@st.cache_resource
def load_guideline_db():
    return compliance_agent.load_guideline_db()

guideline_db = load_guideline_db()

# -----------------------------------------
# IRDAI Compliance Checker Module
# -----------------------------------------
if app_mode == "IRDAI Compliance Checker":
    st.markdown('<div class="section-header">📌 IRDAI Compliance Checker</div>', unsafe_allow_html=True)

    st.markdown("#### 1️⃣ Upload your policy document (PDF)")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="compliance_upload")

    # Handle PDF upload and embed ONCE
    if uploaded_file:
        if (
            "user_policy_db" not in st.session_state or
            st.session_state.get("uploaded_file_name") != uploaded_file.name
        ):
            with st.spinner("🔍 Processing your document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_pdf_path = tmp_file.name

                user_policy_db = compliance_agent.embed_user_policy(tmp_pdf_path)
                os.unlink(tmp_pdf_path)

                st.session_state["user_policy_db"] = user_policy_db
                st.session_state["uploaded_file_name"] = uploaded_file.name

            st.success(f"✅ Uploaded and processed: {uploaded_file.name}")
        else:
            st.success(f"✅ Already processed: {uploaded_file.name}")

        # Choose category
        st.markdown("#### 2️⃣ Choose policy category")
        categories = compliance_agent.get_available_categories()
        selected_category = st.selectbox("Select policy category:", categories)

        if st.button("🚦 Run Compliance Check"):
            with st.spinner("🛡️ Checking compliance..."):
                results, summary = compliance_agent.run_compliance_check(
                    selected_category,
                    st.session_state["user_policy_db"],
                    guideline_db
                )

            # Display results
            st.success(f"✅ Compliance check complete! Score: {summary['compliance_score']:.2f}%")
            st.markdown("---")

            st.markdown(f"**Summary:**")
            st.write(f"Total checks: {summary['total_checks']}")
            st.write(f"Compliant: {summary['compliant']}")
            st.write(f"Compliance Score: {summary['compliance_score']:.2f}%")

            st.markdown("---")
            st.markdown("### 📋 Detailed Results")
            for r in results:
                st.markdown(f"**[{r['id']}] {r['description']}**")
                st.write(f"**Status:** {r['status']}  |  **Similarity Score:** {r['similarity_score']:.2f}")

                with st.expander("🗂️ Official Guideline Snippet"):
                    st.write(r["official_guideline"])

                with st.expander("📜 Your Policy Snippet"):
                    st.write(r["user_policy_text"])

            # Show only up to 5 recommendations
            recommendations = [r for r in results if r["status"] != "✅ Compliant"]
            if recommendations:
                st.markdown("---")
                st.markdown("### 💡 Top 5 Recommendations")
                for r in recommendations[:5]:
                    st.markdown(f"- **[{r['id']}] {r['description']}**")
                    st.write(f"→ Recommended Action: Add/clarify as per guideline.")
                    st.write(f"→ Guideline Reference: {r['official_guideline'][:200]}...")

    with st.expander("ℹ️ What is IRDAI?"):
        st.markdown("""
            The Insurance Regulatory and Development Authority of India (IRDAI) is the statutory body regulating insurance in India.  
            This module helps ensure your policies and practices comply with IRDAI guidelines.
        """, unsafe_allow_html=True)

# -----------------------------------------
# Policy QA Agent Module
# -----------------------------------------
elif app_mode == "Policy QA Agent":
    st.header("📌 Policy QA Agent")
    st.markdown("""
        Upload your insurance policy document and ask questions about its clauses, coverage, exclusions, and more.
    """)

    # 1️⃣ Upload the PDF
    uploaded_file = st.file_uploader("📄 Upload your policy PDF", type=["pdf"], key="qa_upload")

    if uploaded_file:
        # Only rebuild if this PDF is new
        if (
            "qa_uploaded_file_name" not in st.session_state or
            st.session_state["qa_uploaded_file_name"] != uploaded_file.name
        ):
            with st.spinner("🔄 Processing and indexing your document..."):
                # Save to the uploads folder
                uploads_dir = "Policy_QA_Agent/uploads"
                os.makedirs(uploads_dir, exist_ok=True)
                file_path = os.path.join(uploads_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Rebuild FAISS index
                insurance_qa_agent.rebuild_embeddings_from_upload()

                # Cache uploaded filename
                st.session_state["qa_uploaded_file_name"] = uploaded_file.name

            st.success(f"✅ Uploaded and processed: {uploaded_file.name}")
        else:
            st.success(f"✅ Already processed: {uploaded_file.name}")

        # 2️⃣ Question input
        question = st.text_input("💬 Ask a question about your uploaded policy:")

        if st.button("Get Answer"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("🤖 Thinking..."):
                    answer = insurance_qa_agent.get_answer(question)
                st.markdown("### 🧾 Answer")
                st.write(answer)

    else:
        st.info("Please upload your insurance policy PDF to begin.")

    with st.expander("ℹ️ How this works"):
        st.markdown("""
            1. Upload your insurance policy in PDF format.  
            2. The system will process and index the document.  
            3. Ask specific questions about clauses, coverage, exclusions, etc.  
            4. Get AI-powered answers instantly.  
        """)

# -----------------------------------------
# Footer
# -----------------------------------------
st.markdown("""
    <div class="footer">
        🛡️ Insurance Advisor App | Made with ❤️ using Streamlit by Yash Baviskar for DSW Hackathon
    </div>
""", unsafe_allow_html=True)
