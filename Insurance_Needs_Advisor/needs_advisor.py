from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3
    )
    return llm

# Hardcoded location risk descriptions
LOCATION_RISKS = {
    "Mumbai": "flood-prone coastal area with seismic activity (Zone III)",
    "Delhi": "high air pollution, urban flooding, and seismic risk (Zone IV)",
    "Chennai": "cyclone-prone coastal flood zone in seismic Zone III",
    "Kerala": "flood and landslide-prone high rainfall, seismic Zone III",
    "Kolkata": "cyclone and flood risk in low-elevation coastal area, seismic Zone III",
    "Bangalore": "low natural disaster risk but high urban stress and occasional flooding",
    "Hyderabad": "flash flooding from heavy monsoon rains due to poor drainage",
    "Pune": "urban flooding risks from monsoon rains and hill‑slope terrain",
    "Ahmedabad": "acute heatwaves, drought-prone semi‑arid region",
    "Guwahati": "very high vulnerability to earthquakes, floods, and storms",
    "Lucknow": "flood risk from monsoon and moderate seismic activity (Zone III)",
    "Patna": "recurrent flooding from Himalayan river systems"
}

def get_location_risk(location):
    # Map city name to description, fallback to generic
    return LOCATION_RISKS.get(location.strip(), f"general {location} area with typical risks")

# Template for generating recommendations
PROMPT_TEMPLATE = """
You are an expert insurance advisor.

Given the following user profile, recommend the most suitable insurance products and explain why.

Be specific and clear. If any input is missing, proceed with available data.

**Instructions:**
- Recommend insurance product TYPES (e.g. term life, health insurance, motor insurance). do not say example say product name
- For each recommended type, also suggest an example real product name available in India (e.g. Max Life Smart Term Plan, HDFC Ergo Health Suraksha, ICICI Lombard Private Car Insurance).
- Justify each recommendation based on the user profile.

**User Profile:**

- Age: {age}
- Dependents: {dependents}
- Income: {income}
- Assets: {assets}
- Health Conditions: {health_conditions}
- Location Risk: {location}

**Response format:**

Recommendation:
<Your detailed recommendation with product types and example product names here>

Justification:
<Your justification here>
Scores:
[
  "life": <float between 0 and 1>,
  "health": <float between 0 and 1>,
  "motor": <float between 0 and 1>
]
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=[
        "age",
        "dependents",
        "income",
        "assets",
        "health_conditions",
        "location"
    ]
)

llm = load_llm()

def get_insurance_recommendation(
    age,
    dependents,
    income,
    assets,
    health_conditions,
    location
):
    location_risk = get_location_risk(location)

    chain = prompt | llm
    response = chain.invoke({
        "age": age,
        "dependents": dependents,
        "income": income,
        "assets": assets,
        "health_conditions": health_conditions,
        "location": location_risk
    })
    return response.content
