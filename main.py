import streamlit as st
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from io import BytesIO

# Load the service account credentials from Streamlit secrets
service_account_info = {
    "type": st.secrets["gcp"]["type"],
    "project_id": st.secrets["gcp"]["project_id"],
    "private_key_id": st.secrets["gcp"]["private_key_id"],
    "private_key": st.secrets["gcp"]["private_key"],
    "client_email": st.secrets["gcp"]["client_email"],
    "client_id": st.secrets["gcp"]["client_id"],
    "auth_uri": st.secrets["gcp"]["auth_uri"],
    "token_uri": st.secrets["gcp"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcp"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcp"]["client_x509_cert_url"]
}

# Create credentials object from the service account info
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Initialize the Vertex AI SDK with the credentials
vertexai.init(project=service_account_info["project_id"], location="us-central1", credentials=credentials)

# Load the model
model = GenerativeModel("gemini-1.5-pro-preview-0409")

# Set up the Streamlit app
st.title('Chat with Gemini')

user_input = st.text_input("What is up?")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if user_input or uploaded_file:
    with st.chat_message("user"):
        st.markdown(user_input)
        if uploaded_file:
            st.image(uploaded_file, width=200)  # Adjust the width as needed

    with st.chat_message("assistant"):
        with st.spinner('Waiting for the assistant to respond...'):
            # Prepare the contents list
            contents = [user_input]

            if uploaded_file:
                image_bytes = BytesIO(uploaded_file.read())
                image_file = Part.from_bytes(image_bytes.getvalue(), mime_type=uploaded_file.type)
                contents.append(image_file)

            response = model.generate_content(contents)

            if isinstance(response, str):
                st.error(response)
            else:
                # Extract the text value from the response
                response_text = response.text
                st.markdown(response_text)
