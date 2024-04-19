import streamlit as st
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Initialize Streamlit app
st.title("Multimodal Gemini 1.5 Pro Interface")

# Load the service account credentials
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp"])

# Initialize the Vertex AI SDK
vertexai.init(project=st.secrets["gcp"]["project_id"], location="us-central1", credentials=credentials)

# Load the model
model = GenerativeModel("gemini-1.5-pro-preview-0409")

# Text prompt for context and questions
user_prompt = st.text_input("Ask a question or describe your request")

# File uploaders for video, image, and PDF
uploaded_files = st.file_uploader(
    "Upload Video, Image, or PDF", 
    type=['mp4', 'png', 'jpg', 'jpeg', 'pdf'], 
    accept_multiple_files=True
)

# Process and generate content when button is clicked
if st.button('Generate Content'):
    contents = []
    
    # Add the uploaded files to contents if any
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            mime_type = "application/pdf" if uploaded_file.type == "application/pdf" else uploaded_file.type
            part = Part.from_file(uploaded_file, mime_type=mime_type)
            contents.append(part)
            
    # Add the user prompt to contents
    if user_prompt:
        contents.append(user_prompt)
    
    if contents:  # Check if there is anything to process
        response = model.generate_content(contents)
        st.write(response.text)
    else:
        st.error("Please upload a file and/or enter a text prompt.")
