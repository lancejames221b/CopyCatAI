import google.generativeai as genai
import json
import os
import configparser
from requests.exceptions import RequestException, Timeout
import uuid

def guid_generator():
    return str(uuid.uuid4())

TIMEOUT_SECONDS = 60

home_dir = os.path.expanduser("~")
bundle_dir = os.path.join(home_dir, "Library", "Application Support", "CopyCat")
models_path = os.path.join(bundle_dir, "models.json")

class GoogleAPI:
    def __init__(self, api_key, config_file=None):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.config_file = config_file
        
    def generate_response(self, messages, model, max_tokens=None, temperature=0.8):
        """
        Generate a response using Google's Gemini models.
        
        Args:
            messages: List of message objects with role and content
            model: The Gemini model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for response generation
            
        Returns:
            Response text, prompt tokens, completion tokens, and total tokens
        """
        try:
            # Convert messages to Gemini format
            system_prompt = None
            gemini_messages = []
            
            for message in messages:
                if message["role"] == "system":
                    system_prompt = message["content"]
                else:
                    gemini_messages.append({
                        "role": "user" if message["role"] == "user" else "model",
                        "parts": [{"text": message["content"]}]
                    })
            
            # If no max_tokens specified, use a default
            if not max_tokens:
                with open(models_path, "r") as f:
                    models = json.load(f)
                if model in models:
                    max_tokens = int(models[model]["token_size"] * 0.9)  # 90% of max
                else:
                    max_tokens = 4000  # Default fallback
            
            # Initialize the model
            gemini_model = genai.GenerativeModel(model_name=model)
            
            # Add system prompt if available
            if system_prompt:
                gemini_messages.insert(0, {
                    "role": "user",
                    "parts": [{"text": f"System: {system_prompt}"}]
                })
            
            # Create the chat session
            chat = gemini_model.start_chat(history=gemini_messages)
            
            # Generate response
            response = chat.send_message(
                gemini_messages[-1]["parts"][0]["text"],
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            
            # Extract response content
            response_text = response.text
            
            # Estimate token usage (Gemini doesn't provide token counts directly)
            # Rough estimate: 4 chars = 1 token
            prompt_text = "".join([msg["parts"][0]["text"] for msg in gemini_messages])
            prompt_tokens = len(prompt_text) // 4
            completion_tokens = len(response_text) // 4
            total_tokens = prompt_tokens + completion_tokens
            
            return response_text, prompt_tokens, completion_tokens, total_tokens
            
        except Exception as e:
            print(f"Google API Error: {str(e)}")
            raise e

def calculate_google_cost(prompt_tokens, completion_tokens, model=None):
    """Calculate the cost of a Google Gemini API request."""
    with open(models_path, "r") as f:
        models = json.load(f)
    
    if model in models:
        input_price_per_token = models[model]["input_price_per_1k_tokens"] / 1000
        output_price_per_token = models[model]["output_price_per_1k_tokens"] / 1000
        total_price = (prompt_tokens * input_price_per_token) + (completion_tokens * output_price_per_token)
        return total_price
    else:
        return 0