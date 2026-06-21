import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables (expects GOOGLE_API_KEY in .env)
load_dotenv()

# Initialize the LLM (cached to avoid re‑initialization)
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def process_blood_report(blood_report: str):
    """Run the two‑stage pipeline: extraction → diet plan."""
    llm = get_llm()

    # Stage 1: Extract values with status
    extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL 
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""
    with st.spinner("Extracting blood values..."):
        extraction_response = llm.invoke(extraction_prompt)
        extraction_values = extraction_response.text

    # Stage 2: Generate summary and diet plan
    diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 4-5 lines explaining the patient's condition in simple language
2. A short, practical Indian diet plan having only two sections (1) Foods to avoid (2) Foods to eat more of. 
   Do not include any other sections in diet plan.

Blood Work Analysis:
{extraction_values}
"""
    with st.spinner("Generating health summary and diet plan..."):
        diet_response = llm.invoke(diet_prompt)
        diet_text = diet_response.text

    return extraction_values, diet_text

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Blood Report Analyzer", page_icon="🩸")
st.title("🩸 Blood Report Analyzer & Diet Planner")
st.markdown("Paste your blood report below. The app will extract values, flag abnormalities, and suggest an Indian diet plan.")

# Input
blood_report = st.text_area(
    "Enter your blood report (text format):",
    height=300,
    placeholder="e.g., Hemoglobin: 14.5 g/dL (Ref: 13-17)\nWBC: 7.2 x 10^3/µL (Ref: 4-11)\n..."
)

if st.button("Analyze Report", type="primary"):
    if not blood_report.strip():
        st.error("Please paste a blood report.")
    else:
        try:
            extraction_values, diet_text = process_blood_report(blood_report)

            # Display results
            st.subheader("📊 Extracted Values & Status")
            st.markdown(extraction_values)

            st.subheader("📋 Health Summary & Diet Plan")
            st.markdown(diet_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Make sure your Google API key is set in a `.env` file or as an environment variable.")

# Helpful notes
with st.expander("ℹ️ How to set up your Google API key"):
    st.markdown("""
    1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey).
    2. Create a `.env` file in the same folder as this app with:
    GOOGLE_API_KEY=your_api_key_here
    3. Or set it as an environment variable before running the app.
""")