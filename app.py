import os
from PIL import Image
from phi.agent import Agent
from phi.model.google import Gemini
import streamlit as st
import requests

# Function to perform Google Custom Search
def google_custom_search(query, api_key, cse_id, num_results=3):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results
    }
    response = requests.get(search_url, params=params)
    response.raise_for_status()  # Raises HTTPError for bad responses
    return response.json().get("items", [])

# Streamlit sidebar configuration for API keys
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state.GEMINI_API_KEY = None

with st.sidebar:
    st.title("ℹ️ Configuration")

    # User inputs Gemini API Key
    if not st.session_state.GEMINI_API_KEY:
        gemini_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password"
        )
        st.caption(
            "Get your API key from [Google AI Studio](https://aistudio.google.com/apikey) 🔑"
        )
        if gemini_key:
            st.session_state.GEMINI_API_KEY = gemini_key
            st.success("Gemini API Key saved!")
            st.rerun()
    else:
        st.success("Gemini API Key is configured")
        if st.button("🔄 Reset Gemini API Key"):
            st.session_state.GEMINI_API_KEY = None
            st.experimental_rerun()

    # Check for Serper API Key in secrets
    if "SERPER_API_KEY" not in st.secrets:
        st.error("Please add your Serper API Key to the Streamlit secrets.")

    st.info(
        "This tool provides AI-powered analysis of medical imaging data using "
        "advanced computer vision and radiological expertise."
    )
    st.warning(
        "⚠DISCLAIMER: This tool is for educational and informational purposes only. "
        "All analyses should be reviewed by qualified healthcare professionals. "
        "Do not make medical decisions based solely on this analysis."
    )

# Initialize Medical Agent if both keys are available
medical_agent = None
if st.session_state.GEMINI_API_KEY and "SERPER_API_KEY" in st.secrets:
    medical_agent = Agent(
        model=Gemini(
            api_key=st.session_state.GEMINI_API_KEY,
            id="gemini-2.0-flash-exp"
        ),
        tools=None,
        markdown=True
    )

if not medical_agent:
    st.warning("Please configure both the Gemini and Serper API keys to continue.")

# Medical Analysis Query
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the GoogleSearch tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

# App UI
st.title("🏥 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for professional analysis")

upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))

            st.image(
                resized_image,
                caption="Uploaded Medical Image",
                use_column_width=True
            )

            analyze_button = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            )

    with analysis_container:
        if analyze_button:
            image_path = "temp_medical_image.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    response = medical_agent.run(query, images=[image_path])
                    st.markdown("### 📋 Analysis Results")
                    st.markdown("---")
                    st.markdown(response.content)
                    st.markdown("---")
                    st.caption(
                        "Note: This analysis is generated by AI and should be reviewed by "
                        "a qualified healthcare professional."
                    )

                    # Perform Google Custom Search
                    search_results = google_custom_search(
                        query="Medical imaging abnormalities similar cases",
                        api_key=st.secrets["SERPER_API_KEY"],
                        cse_id="custom-search-engine-id"
                    )
                    st.markdown("### 🔗 Relevant Medical Literature")
                    for result in search_results:
                        st.markdown(f"**[{result['title']}]({result['link']})**")
                        st.write(result['snippet'])
                        st.markdown("---")

                except Exception as e:
                    st.error(f"Analysis error: {e}")
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)
else:
    st.info("👆 Please upload a medical image to begin analysis.")
