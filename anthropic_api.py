import anthropic
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

class AnthropicAPI:
    def __init__(self, api_key, config_file=None):
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self.config_file = config_file
        
    def generate_response(self, messages, model, max_tokens=None, temperature=0.8):
        """
        Generate a response using Anthropic's Claude models.
        
        Args:
            messages: List of message objects with role and content
            model: The Claude model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for response generation
            
        Returns:
            Response text, prompt tokens, completion tokens, and total tokens
        """
        try:
            # Convert messages to Anthropic format
            system_prompt = None
            anthropic_messages = []
            
            for message in messages:
                if message["role"] == "system":
                    system_prompt = message["content"]
                else:
                    anthropic_messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
            
            # If no max_tokens specified, use a default
            if not max_tokens:
                with open(models_path, "r") as f:
                    models = json.load(f)
                if model in models:
                    max_tokens = int(models[model]["token_size"] * 0.9)  # 90% of max
                else:
                    max_tokens = 4000  # Default fallback
            
            # Create the message for Claude
            response = self.client.messages.create(
                model=model,
                messages=anthropic_messages,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response content
            response_text = response.content[0].text
            
            # Get token usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            
            return response_text, prompt_tokens, completion_tokens, total_tokens
            
        except Exception as e:
            print(f"Anthropic API Error: {str(e)}")
            raise e

def calculate_anthropic_cost(prompt_tokens, completion_tokens, model=None):
    """Calculate the cost of an Anthropic API request."""
    with open(models_path, "r") as f:
        models = json.load(f)
    
    if model in models:
        input_price_per_token = models[model]["input_price_per_1k_tokens"] / 1000
        output_price_per_token = models[model]["output_price_per_1k_tokens"] / 1000
        total_price = (prompt_tokens * input_price_per_token) + (completion_tokens * output_price_per_token)
        return total_price
    else:
        return 0