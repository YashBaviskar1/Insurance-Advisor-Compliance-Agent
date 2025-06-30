import streamlit as st
from IRDAI_compliance_agent import compliance_agent
from Policy_QA_Agent import insurance_qa_agent 
import tempfile
import os
from Insurance_Needs_Advisor import needs_advisor
from Insurance_Needs_Advisor.insurance_recommender import InsuranceRecommender
from Policy_Coverage_Agent.gap_coverage import ingest_policies_from_directory, load_policy_store, get_gap_recommendations

import re
import json
import ast
recommender = InsuranceRecommender("Insurance_Needs_Advisor/policies.csv")
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
        ["IRDAI Compliance Checker", "Policy QA Agent", "Insurance Need Advisor", "Policy Coverage Agent"]
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
elif app_mode == "Insurance Need Advisor":
    st.markdown('<div class="section-header">📌 Insurance Needs Advisor</div>', unsafe_allow_html=True)

    st.markdown("""
        Fill in your details below to get personalized insurance recommendations
        with real example products and analysis.
    """)

    # User profile input form
    with st.form("advisor_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Annual Income (INR)", min_value=0, step=50000, value=500000)
            assets = st.text_input("Key Assets (comma-separated)", value="car, house")
        with col2:
            dependents = st.number_input("Number of Dependents", min_value=0, value=2)
            health_conditions = st.text_area("Health Conditions (if any)", value="none")
            location = st.text_input("City / Location", value="Mumbai")

        needs = st.multiselect(
            "Insurance Types you want to consider", 
            ["Health", "Term Life", "Accident", "Life", "Accident/Term Life", "Life/Accident", "Health/Maternity"], 
            default=["Health", "Term Life"]
        )
        
        max_premium = st.number_input("Maximum Annual Premium (INR)", min_value=0, value=2000, step=500)

        submitted = st.form_submit_button("🛡️ Get Recommendation")

    if submitted:
        with st.spinner("🤖 Analyzing your profile and generating recommendations..."):
            user_profile = {
                "age": age,
                "income": income,
                "needs": needs,
                "max_premium": max_premium
            }

        recommendations = recommender.recommend(user_profile)

        advice_text = needs_advisor.get_insurance_recommendation(
            age=age,
            dependents=dependents,
            income=income,
            assets=assets,
            health_conditions=health_conditions,
            location=location
        )

    # ✅ Show the LLM *advice text* FIRST
        st.success("✅ Personalized Insurance Needs Analysis & Advice")
        st.markdown(advice_text)
        import re
        import json

        def extract_scores(text):
            match = re.search(r"Scores:\s*(\[.*\])", text, re.DOTALL)
            if match:
                raw = match.group(1)
                # Heuristic fix for broken format
                if ":" in raw and "{" not in raw:
                    # Convert: [ "a": 0.8, "b": 0.9 ]  -> [ { "a": 0.8, "b": 0.9 } ]
                    raw = "[{" + raw.strip()[1:-1] + "}]"
                try:
                    return ast.literal_eval(raw)
                except Exception:
                    return None
            return None
        scores = extract_scores(advice_text)

        # 4️⃣ Show coverage adequacy progress bars
        if scores:
            st.subheader("📊 Coverage Adequacy")
            for coverage_type, score in scores:
                coverage_label = coverage_type.capitalize() + " Coverage"
                st.progress(score, text=f"{coverage_label}: {int(score * 100)}% adequate")
        else:
                st.info("⚠️ No coverage adequacy scores were found in the recommendation. Please review the recommendation text.")
        if recommendations.empty:
            st.warning("⚠️ Sorry! No matching policies were found for your profile.")
        else:
            st.subheader("📋 Recommended Insurance Policies")
            st.dataframe(
                recommendations[[
                    "Name", "Type", "Annual Premium", "Sum Assured", "Eligibility Notes", "Score"
                ]].reset_index(drop=True),
                use_container_width=True
            )
# -----------------------------------------
# Policy Coverage Agent Module
# -----------------------------------------

elif app_mode == "Policy Coverage Agent":
    st.markdown('<div class="section-header">📌 Policy Coverage Gap Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
        Provide your personal and risk profile details below to analyze potential coverage gaps and receive recommendations.
        You can also upload your existing policy PDFs (optional) to include them in the analysis.
    """)
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        st.subheader("Optional: Upload Your Existing Policy PDFs")
    uploaded_pdfs = st.file_uploader(
            "Upload one or more PDF files (optional)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
    new_uploads = [f for f in uploaded_pdfs if f not in st.session_state.uploaded_files]

    if new_uploads:
        with st.spinner("Saving and ingesting your uploaded policies..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for pdf_file in new_uploads:
                    file_path = os.path.join(tmp_dir, pdf_file.name)
                    with open(file_path, "wb") as f:
                        f.write(pdf_file.read())
                
                try:
                    num_ingested = ingest_policies_from_directory(tmp_dir)
                    st.success(f"✅ Ingested {num_ingested} new policy sections")
                    # Update session state
                    st.session_state.uploaded_files.extend(new_uploads)
                except Exception as e:
                    st.error(f"❌ Error ingesting policies: {e}")

    st.subheader("User Profile Inputs")
    age = st.number_input("Age", min_value=0, max_value=120, value=22)
    num_dep = st.number_input("Number of Dependents", min_value=0, max_value=10, value=3)
    dependents = []
    for i in range(num_dep):
        st.write(f"Dependent #{i+1}")
        relation = st.text_input(f"Relation {i+1}", value="parent", key=f"rel{i}")
        dep_age = st.number_input(f"Age {i+1}", min_value=0, max_value=120, value=50, key=f"age{i}")
        dependents.append({"relation": relation, "age": dep_age})

    income = st.text_input("Income (e.g. 500000)", "500000")
    assets = st.text_input("Assets (comma-separated)", "scooter").split(",")
    health_conditions = st.text_input("Health Conditions (comma-separated)", "").split(",")
    location = st.selectbox("Location", [
        "Mumbai","Delhi","Chennai","Kerala","Kolkata",
        "Bangalore","Hyderabad","Pune","Ahmedabad","Guwahati","Lucknow","Patna"
    ])
    occupation = st.text_input("Occupation", "Student")

    st.subheader("Risk Factors (auto-derived)")
    flood_zone = st.checkbox("Flood Zone", value=True)
    earthquake_zone = st.slider("Earthquake Zone (1-5)", 1, 5, 3)
    high_pollution = st.checkbox("High Pollution", value=False)

    if st.button("Analyze Coverage Gaps"):
        user_profile = {
            "age": age,
            "dependents": dependents,
            "income": income,
            "assets": [a.strip() for a in assets if a.strip()],
            "health_conditions": [h.strip() for h in health_conditions if h.strip()],
            "location": location,
            "occupation": occupation,
            "risk_factors": {
                "flood_zone": flood_zone,
                "earthquake_zone": earthquake_zone,
                "high_pollution": high_pollution
            }
        }

        with st.spinner("🔎 Loading policy database..."):
            policy_store = load_policy_store()

        try:
            result = get_gap_recommendations(user_profile, policy_store)
            # Try parsing if string
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    try:
                        result = ast.literal_eval(result)
                    except Exception:
                        st.error("❌ Could not parse result to JSON or Python dict.")
                        result = None
        except ValueError as e:
            st.error(f"❌ Error in analysis: {e}")
            result = None

        if result:
            try:
                st.subheader(f"Risk Score: {result.get('risk_score',0)*100:.0f}/100")
                st.markdown("---")

                st.subheader("Rule-based Gaps")
                for gap in result.get('rule_based_gaps', []):
                    st.error(gap)

                st.subheader("Inference-Agent based Rules")
                for gap in result.get('llm_gaps', []):
                    st.warning(gap)

                st.subheader(" Priority Gaps")
                for gap in result.get('priority_gaps', []):
                    st.success(gap)

                st.subheader("Recommendations")
                for rec in result.get('recommendations', []):
                    if isinstance(rec, dict):
                        st.markdown(f"**{rec.get('type','').title()}** — *{rec.get('example','')}*\n> {rec.get('justification','')}")
                    else:
                        st.markdown(f"- {rec}")

                st.balloons()
            except Exception as e:
                st.error(f"❌ Error displaying results: {e}")
        else:
            st.warning("⚠️ No valid result returned. Please check your inputs or try again later.")

# -----------------------------------------
# Footer
# -----------------------------------------
st.markdown("""
    <div class="footer">
        🛡️ Insurance Advisor App | Made with ❤️ using Streamlit by Yash Baviskar for DSW Hackathon
    </div>
""", unsafe_allow_html=True)
