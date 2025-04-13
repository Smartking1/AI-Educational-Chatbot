import os
import streamlit as st
import tempfile
import traceback
import nest_asyncio
from typing import List
from llama_index.core import VectorStoreIndex, Document, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import PyPDF2
from utils.model import LLMClient
import logging
import uuid

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Set HuggingFace embedding model
hf_embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define a prompt template to structure the query
prompt_template = PromptTemplate(
    input_variables="query",
    template="You are Bob, an AI assistant. Answer the following question concisely and clearly with only the answer: {query}"
)

# In-memory storage for indices using session ID
index_storage = {}

# Function to extract text from uploaded PDFs
def extract_text_from_pdfs(uploaded_files):
    document_text = ""
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                document_text += page.extract_text()
        os.remove(tmp_path)
    return document_text

# Function to process uploaded files and generate vector index
def process_files(uploaded_files):
    document_text = extract_text_from_pdfs(uploaded_files)
    document = Document(text=document_text)
    index = VectorStoreIndex.from_documents([document], embed_model=hf_embedding_model)
    session_id = str(uuid.uuid4())
    index_storage[session_id] = index
    return session_id

# Function to answer questions using the indexed data
def qa_engine(query: str, index: VectorStoreIndex, llm_client, choice_k=3):
    query_engine = index.as_query_engine(
        llm=llm_client,
        similarity_top_k=choice_k,
        verbose=True,
        Streaming=True,
        prompt_template=prompt_template
    )
    response = query_engine.query(query)
    answer = response.response.split("Answer:")[-1].strip()
    return answer, response

# Streamlit UI starts here
st.title("📄 PDF Chat with AI")

uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        session_id = process_files(uploaded_files)
    st.success(f"Files processed. Session ID: {session_id}")
    st.session_state["session_id"] = session_id

    question = st.text_input("Ask a question based on the uploaded documents")
    selected_model = st.selectbox("Choose a model", [ "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "claude-3-5-sonnet",
        "claude-3-haiku",
        "gemini-1.5-flash",
        "gemini-1.5-pro",])

    if st.button("Generate Answer") and question:
        try:
            init_client = LLMClient(
                groq_api_key=st.secrets.get("GROQ_API_KEY"),
                gcp_credentials=st.secrets.get("gcp_service_account"),
                #secrets_path=None,  # Not using external path, reading from secrets.toml directly
                temperature=0.7,
                max_output_tokens=512
            )
            llm_client = init_client.map_client_to_model(selected_model)
            index = index_storage[st.session_state["session_id"]]
            answer, response = qa_engine(question, index=index, llm_client=llm_client, choice_k=3)
            st.markdown("**Answer:**")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.text(traceback.format_exc())

st.sidebar.title("🛠 App Info")
st.sidebar.markdown("This app allows you to chat with your uploaded PDFs using various LLMs.")
