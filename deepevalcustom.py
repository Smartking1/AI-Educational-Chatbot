import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models import DeepEvalBaseLLM
# Custom class inheriting from DeepEvalBaseLLM
class CustomLlama3_8B(DeepEvalBaseLLM):
    def __init__(self):
        # Load the model without bitsandbytes config for CPU usage
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto"  # Automatically maps to CPU
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        # Define the pipeline for text generation using the model and tokenizer
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",  # Automatically maps to CPU
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Generate text based on the provided prompt
        return pipeline(prompt)[0]["generated_text"]

    async def a_generate(self, prompt: str) -> str:
        # Asynchronous version of the generate method
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3 8B"

# Example usage in main.py
if __name__ == "__main__":
    custom_model = CustomLlama3_8B()
