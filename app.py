import streamlit as st

# Page metadata
st.set_page_config(
    page_title="Insurance Advisor and Compliance Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to style the app
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

# Header
st.markdown('<div class="main-title">🛡️ Insurance Advisor and Compliance Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An AI-powered assistant to help you choose the right insurance, understand policies, detect gaps, and stay IRDAI compliant.</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("🗂️ Navigation")
    app_mode = st.radio(
        "Choose a module:",
        ["IRDAI Compliance Checker", "Policy QA Agent"]
    )
    st.markdown("---")
    st.markdown("**ℹ️ About this app**")
    st.caption("This AI tool assists users in insurance advisory, policy question-answering, and IRDAI compliance checking.")

# Main area
if app_mode == "IRDAI Compliance Checker":
    st.markdown('<div class="section-header">📌 IRDAI Compliance Checker</div>', unsafe_allow_html=True)
    st.file_uploader("📄 Upload your compliance document (PDF)", type=["pdf"])
    with st.expander("ℹ️ What is IRDAI?"):
        st.markdown("""
            The Insurance Regulatory and Development Authority of India (IRDAI) is the statutory body regulating insurance in India.  
            This module helps ensure your policies and practices comply with IRDAI guidelines.
        """, unsafe_allow_html=True)
elif app_mode == "Policy QA Agent":
    st.header("📌 Policy QA Agent")
    st.markdown("""
        Upload your insurance policy document and ask questions about its clauses, coverage, exclusions, and more.
    """)

    with st.container():
        uploaded_file = st.file_uploader("📄 Upload your policy PDF", type=["pdf"])
        question = st.text_input("💬 Ask a question about your uploaded policy:")
        if st.button("Get Answer"):
            if uploaded_file is None:
                st.warning("Please upload a PDF document first.")
            else:
                st.success("Answer will be displayed here. (Functionality placeholder)")

    with st.expander("ℹ️ How this works"):
        st.markdown("""
            1. Upload your insurance policy in PDF format.  
            2. Ask specific questions about clauses, coverage, exclusions, etc.  
            3. Get AI-powered answers instantly.  
        """)

# Footer
st.markdown("""
    <div class="footer">
        🛡️ Insurance Advisor App | Made with ❤️ using Streamlit by Yash Baviskar for DSW Hackathon
    </div>
""", unsafe_allow_html=True)
