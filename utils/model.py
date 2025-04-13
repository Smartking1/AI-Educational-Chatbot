import json
from pydantic import BaseModel
from llama_index.llms.groq import Groq
from llama_index.llms.vertex import Vertex
from llama_index.llms.anthropic import Anthropic
from anthropic import AnthropicVertex
from google.oauth2 import service_account

class LLMClient(BaseModel):
    groq_api_key: str = ""
    gcp_credentials: dict = None  # ✅ Accept as dictionary now
    temperature: float = 0.1
    max_output_tokens: int = 512

    def load_credentials(self):
        if not self.gcp_credentials:
            raise ValueError("GCP credentials not provided")

        credentials = service_account.Credentials.from_service_account_info(
            self.gcp_credentials,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        return credentials

    def refresh_auth(self, credentials):
        from google.auth.transport.requests import Request
        credentials.refresh(Request())
        return credentials

    def generate_access_token(self, credentials):
        _credentials = self.refresh_auth(credentials)
        access_token = _credentials.token
        if not access_token:
            raise RuntimeError("Could not resolve API token from the environment")
        return access_token

    def groq(self, model):
        return Groq(
            model=model,
            api_key=self.groq_api_key,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens
        )

    def gemini(self, model):
        credentials = self.load_credentials()
        return Vertex(
            model=model,
            project=credentials.project_id,
            credentials=credentials,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens
        )

    def anthropic(self, model):
        credentials = self.load_credentials()
        access_token = self.generate_access_token(credentials)

        region_mapping = {
            "claude-3-5-sonnet@20240620": "us-east5",
            "claude-3-haiku@20240307": "us-central1",
            "claude-3-opus@20240229": "us-central1",
        }

        vertex_client = AnthropicVertex(
            access_token=access_token,
            project_id=credentials.project_id,
            region=region_mapping.get(model)
        )

        return Anthropic(
            model=model,
            vertex_client=vertex_client,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens
        )

    def map_client_to_model(self, model):
        model_mapping = {
            "llama-3.1-70b-versatile": self.groq,
            "llama-3.1-8b-instant": self.groq,
            "mixtral-8x7b-32768": self.groq,
            "claude-3-5-sonnet@20240620": self.anthropic,
            "claude-3-haiku@20240307": self.anthropic,
            "claude-3-3-opus@20240229": self.anthropic,
            "gemini-1.5-flash": self.gemini,
            "gemini-1.5-pro": self.gemini,
        }

        _client = model_mapping.get(model)
        if not _client:
            raise ValueError(f"Unsupported model: {model}")
        return _client(model)
