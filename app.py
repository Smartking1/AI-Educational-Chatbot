import os
import tempfile
import traceback
import nest_asyncio  # Import nest_asyncio to allow nested async
from typing import List
from fastapi import FastAPI, Form, UploadFile, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import PyPDF2
from utils.model import LLMClient
import logging
import uuid

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Set HuggingFace embedding model
hf_embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define a prompt template to structure the query
prompt_template = PromptTemplate(
    input_variables="query",
    template="You are Bob, an AI assistant. Answer the following question concisely and clearly with only the answer: {query}"
)

# Initialize FastAPI app
app = FastAPI()

# In-memory storage for indices using session ID
index_storage = {}

# Define the qa_engine function
def qa_engine(query: str, index: VectorStoreIndex, llm_client, choice_k=3):
    # Create a query engine from the index
    query_engine = index.as_query_engine(
        llm=llm_client,
        similarity_top_k=choice_k,
        verbose=True,
        Streaming=True,
        prompt_template=prompt_template
    )
    
    # Process the query
    response = query_engine.query(query)
    
    # Extract the final answer from the response
    answer = response.response.split("Answer:")[-1].strip()
    
    # Return both answer and response for logging or further evaluation
    return answer, response

@app.get('/healthz')
async def health():
    return {
        "application": "Simple LLM API",
        "message": "running successfully"
    }

@app.post('/upload')
async def process(
    files: List[UploadFile] = None
):
    try:
        # Generate a session ID to store the index
        session_id = str(uuid.uuid4())  # Generate a UUID to serve as the session ID
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded files temporarily
            if files:
                for file in files:
                    with open(os.path.join(temp_dir, file.filename), "wb") as buffer:
                        buffer.write(await file.read())

            # Extract text from PDF files
            document_text = ""
            for filename in os.listdir(temp_dir):
                if filename.endswith(".pdf"):
                    with open(os.path.join(temp_dir, filename), "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            document_text += page.extract_text()

            # Create Document object in-memory
            document = Document(text=document_text)

            # Create VectorStoreIndex from Document, explicitly using the HuggingFace embedding model
            index = VectorStoreIndex.from_documents([document], embed_model=hf_embedding_model)

            # Store the index in the in-memory storage with session_id as the key
            index_storage[session_id] = index

            return {
                "detail": "Processing Done",
                "status_code": 200,
                "session_id": session_id  # Return the session ID to the frontend
            }
    except Exception as e:
        print(traceback.format_exc())
        return {
            "detail": f"Could not generate embeddings: {e}",
            "status_code": 500
        }

@app.post('/generate')
async def generate_chat(request: Request):
    query = await request.json()
    model = query["model"]  # Model selected by the user in the frontend
    question = query["question"]
    session_id = query.get("session_id")  # The session ID passed from the frontend

    if not session_id or session_id not in index_storage:
        return PlainTextResponse(content="Index for the session is missing or invalid.", status_code=400)

    try:
        # Initialize LLM client with the selected model
        init_client = LLMClient(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            secrets_path="C:/Users/DELL PC/EduBot/edubot/EduBot/edubot/gemini.json",
            temperature=0.7,  # Set a default temperature or adjust if needed
            max_output_tokens=512
        )

        # Map to the chosen model
        llm_client = init_client.map_client_to_model(model)  # Now using the selected model from the frontend

        # Retrieve the index from the in-memory storage using the session ID
        index = index_storage[session_id]

        # Process the query using the qa_engine function
        answer, response = qa_engine(
            question,
            index=index,
            llm_client=llm_client,
            choice_k=3
        )

        # Log the query and response
        logging.info(f"Query: {question}")
        logging.info(f"Response: {response.response}")

        # Return the response in a chat-friendly format
        return PlainTextResponse(content=response.response, status_code=200)

    except Exception as e:
        message = f"An error occurred while {model} was trying to generate a response: {e}"
        print(traceback.format_exc())
        return PlainTextResponse(content=message, status_code=500)

if __name__ == "__main__":
    import uvicorn
    print("Starting LLM API")
    uvicorn.run(app, host="0.0.0.0", reload=True)
