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

# File uploaders for video and image
video_file = st.file_uploader("Upload Video", type=['mp4'])
image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

# Text prompt for context and questions
prompt = st.text_area("Enter your questions and context here", """
Watch each frame in the video carefully and answer the questions.
Only base your answers strictly on what information is available in the video attached.
Do not make up any information that is not part of the video and do not be too verbose, be to the point.

Questions:
- When is the moment in the image happening in the video? Provide a timestamp.
- What is the context of the moment and what does the narrator say about it?
""")

# Process and generate content when button is clicked
if st.button('Generate Content'):
    if video_file and image_file:
        video_part = Part.from_file(video_file, mime_type="video/mp4")
        image_part = Part.from_file(image_file, mime_type="image/png")
        contents = [video_part, image_part, prompt]
        response = model.generate_content(contents)
        st.write(response.text)
    else:
        st.error("Please upload both video and image files.")
