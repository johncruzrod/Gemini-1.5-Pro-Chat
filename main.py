import streamlit as st
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

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

# Set up the generation configuration
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

# Set up the safety settings
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Set up the Streamlit app
st.title('Chat with Gemini')

if 'chat' not in st.session_state:
    st.session_state.chat = model.start_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display each message
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"], width=100)
        st.markdown(message["content"])

# User text input
user_input = st.chat_input("What is up?")

# Image upload handling
uploaded_file = st.file_uploader("Upload an image (optional)", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    st.session_state.messages.append({"role": "user", "content": user_input, "image": uploaded_file})

# Handle user input and image
if user_input or uploaded_file:
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
    if uploaded_file:
        with st.chat_message("user"):
            st.image(uploaded_file, width=100)

    # Send to model and display response
    with st.chat_message("assistant"):
        with st.spinner('Waiting for the assistant to respond...'):
            # Convert the conversation history into a list of strings
            conversation_history = [f"{message['role']}: {message['content']}" for message in st.session_state.messages]

            response = st.session_state.chat.send_message(
                conversation_history,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            if isinstance(response, str):
                st.error(response)
            else:
                # Extract the text value from the response
                response_text = response.text
                st.markdown(response_text)
                # Append only the assistant's response to the messages list
                st.session_state.messages.append({"role": "assistant", "content": response_text})
