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
st.title("Vertex AI Conversational Agent")

# Initialize the chat session
if 'chat' not in st.session_state:
    st.session_state.chat = model.start_chat()

# Initialize the conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Get the user input
user_input = st.text_input("Enter your text:")

# Handle the conversation when the user clicks the button
if st.button("Send"):
    if user_input:
        try:
            # Add the user's input to the conversation history
            st.session_state.conversation_history.append(user_input)

            # Generate a response from the AI using the conversation history
            response = st.session_state.chat.send_message(
                st.session_state.conversation_history,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # Add the AI's response to the conversation history
            st.session_state.conversation_history.append(response.text)

            # Display the conversation history
            for message in st.session_state.conversation_history:
                st.write(message)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter some text.")
