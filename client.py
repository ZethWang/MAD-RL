from typing import List, Optional, Any, Dict
import requests
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenClient:
    """Client for Qwen2.5-1.5B using the HuggingFace Transformers API"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", device: str = "cuda"):
        self.device = device
        print(f"Loading {model_name} on {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(device)
        print("Model loaded successfully.")
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = 0.8,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response based on the provided messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stop: List of strings that will stop generation when encountered
            
        Returns:
            Dictionary with the completion result
        """
        # Convert messages to the format expected by the tokenizer
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Encode the messages
        inputs = self.tokenizer.apply_chat_template(
            formatted_messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response with consistent parameters
        gen_config = {
            "max_new_tokens": max_tokens or 1024,
        }
        
        # Only add temperature parameters if we're doing sampling
        if temperature > 0:
            gen_config["do_sample"] = True
            gen_config["temperature"] = temperature
        else:
            gen_config["do_sample"] = False  # Greedy decoding
        
        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                **gen_config
            )
        
        # Decode the output
        generated_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Find only the assistant's response (after the last message)
        last_role = messages[-1]["role"]
        if last_role == "user":
            # Need to extract assistant's response
            response_content = generated_output.split("assistant\n")[-1].strip()
        else:
            # Response is already from the assistant
            response_content = generated_output.split(messages[-1]["content"])[-1].strip()
        
        # Calculate token counts
        input_tokens = len(self.tokenizer.encode("\n".join([m["content"] for m in messages])))
        output_tokens = len(self.tokenizer.encode(response_content))
        
        # Create response in the same format as OpenAI's API
        completion = {
            "id": "qwen-generated",
            "object": "chat.completion",
            "created": 0,
            "model": "Qwen/Qwen2.5-1.5B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
        
        return completion


# For backwards compatibility with the original code
LlamaClient = QwenClient  # Use QwenClient in place of LlamaClient

# Usage example
def main():
    # Create client instance
    client = QwenClient(model_name="Qwen/Qwen2.5-1.5B", device="cuda")

    messages = [
            {
                'role': 'system', 
                'content': 'Make sure to state your answer and your confidence at the end of the response following format strictly.'
            },
            {
                'role': 'user', 
                'content': 'What is the result of 52+77*46+60-69*26?'
            }
        ]
    completion = client.create_chat_completion(
        messages=messages, max_tokens=1280, temperature=0
    )
    print("\n=== Chat Completion Example ===")
    print(f'completion\n{json.dumps(completion,indent=2)}')
    print("\n=== Message Example ===")
    print(completion["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
