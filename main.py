import os
import streamlit as st
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Set API URL (assuming FastAPI is hosted locally)
API_URL = "http://localhost:8000"

# List of available models that the user can select from
models = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "claude-3-5-sonnet",
    "claude-3-haiku",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Initialize chat history and session_id in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None  # Store session ID returned from backend

# Function to send a document to the backend API for processing
def send_document_to_api(uploaded_file):
    files = {"files": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    try:
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data['session_id']  # Store session ID in session state
            return data  # Parse the JSON response if it's a success
        else:
            st.error(f"Error {response.status_code}: {response.text}")  # Show error if status is not 200
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to upload the document: {e}")
        return None

# Function to send a query to the backend API for chat generation
def generate_chat(model, question):
    payload = {
        "model": model,
        "question": question,
        "session_id": st.session_state.session_id  # Pass the session ID for index retrieval
    }
    try:
        response = requests.post(f"{API_URL}/generate", json=payload)
        if response.status_code == 200:
            return response.text  # Return the response text if it's a success
        else:
            st.error(f"Error {response.status_code}: {response.text}")  # Handle error cases
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get a response from the backend: {e}")
        return None

# Main Streamlit App
def main():
    st.set_page_config(layout="wide")

    # Sidebar for Model Selection and History
    with st.sidebar:
        st.header("Model Selection & Chat History")

        # Dropdown to allow users to select from the available models
        selected_model = st.selectbox("Select Model", models)

        # History section using session state
        st.subheader("Chat History")
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.write(f"Q: {chat['query']}\nA: {chat['response']}")

    # Main area for Document Upload and Chat Interaction
    st.title("Chat with your AI-Powered Educational Assistant")
    st.subheader("Chat with your Educational Documents")

    uploaded_file = st.file_uploader("Upload document(s)", type=['pdf'])

    if uploaded_file:
        response = send_document_to_api(uploaded_file)
        if response:
            st.success("Documents uploaded and processed successfully!")
        else:
            st.error(f"Error uploading document.")

    # Chat interaction area
    query = st.text_input("Ask a question:")

    if st.button("Send Query") and uploaded_file and query:
        try:
            response = generate_chat(selected_model, query)
            if response:
                # Add new query-response pair to session state history
                st.session_state.chat_history.append({"query": query, "response": response})

                # Display the chat in chat format
                st.subheader("Chat")
                for chat in st.session_state.chat_history:
                    st.write(f"**You**: {chat['query']}")
                    st.write(f"**AI**: {chat['response']}")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
