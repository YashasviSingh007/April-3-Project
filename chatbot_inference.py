import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import random
from collections import deque

class Chatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.response_history = deque(maxlen=5)  # Keep track of last 5 responses
        
    def load_model(self):
        """
        Load the trained model and tokenizer
        """
        print("Loading model and tokenizer...")
        model_path = "./chatbot_model/final"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
    def get_random_prompt(self):
        """
        Get a random prompt template to vary responses
        """
        prompts = [
            "Let me help you with that.",
            "I'd be happy to assist you.",
            "I can help you with that.",
            "I'll do my best to help you.",
            "I'm here to assist you.",
            "I can help you with your request.",
            "I'll help you with that right away.",
            "I'm ready to assist you with that.",
            "I can help you with your question.",
            "I'll be happy to help you with that.",
            "I understand your concern. Let me help.",
            "I'll assist you with that right now.",
            "I'm here to support you with that.",
            "I can definitely help you with that.",
            "Let me assist you with your request.",
            "I'll do everything I can to help you.",
            "I'm ready to help you with that.",
            "I can provide assistance with that.",
            "I'll help you resolve this matter.",
            "I'm here to help you with your needs."
        ]
        return random.choice(prompts)
    
    def is_response_similar(self, new_response, threshold=0.8):
        """
        Check if the new response is too similar to previous responses
        """
        if not self.response_history:
            return False
            
        # Simple similarity check based on word overlap
        new_words = set(new_response.lower().split())
        for old_response in self.response_history:
            old_words = set(old_response.lower().split())
            overlap = len(new_words.intersection(old_words)) / len(new_words)
            if overlap > threshold:
                return True
        return False
    
    def generate_response(self, user_input, max_length=200):
        """
        Generate a response from the model
        """
        # Format the input with more context and random prompt
        prompt = self.get_random_prompt()
        input_text = f"Human: {user_input}\nAssistant: {prompt}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs["input_ids"])
        
        # Try generating responses until we get a unique one
        for _ in range(3):  # Try up to 3 times
            # Generate response with improved parameters
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.9,  # Increased temperature for more variety
                top_p=0.95,
                top_k=60,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.5,  # Increased repetition penalty
                length_penalty=1.3,  # Increased length penalty
                no_repeat_ngram_size=4,  # Prevent repetition of 4-grams
                early_stopping=True,
                num_beams=4  # Increased beam search
            )
            
            # Decode and return the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the Assistant's response
            response = response.split("Assistant:")[-1].strip()
            
            # Check if response is unique
            if not self.is_response_similar(response):
                self.response_history.append(response)
                return response
        
        # If we couldn't generate a unique response, return the last one
        return response

def main():
    try:
        chatbot = Chatbot()
        chatbot.load_model()
        
        # Check if input is provided as command line argument
        if len(sys.argv) > 1:
            user_input = " ".join(sys.argv[1:])
            print("\nAssistant is thinking...")
            response = chatbot.generate_response(user_input)
            print(f"\nAssistant: {response}")
            return

        # Interactive mode
        print("\nChatbot is ready! Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    print("\nPlease enter a message.")
                    continue
                
                # Generate and print response
                print("\nAssistant is thinking...")
                response = chatbot.generate_response(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()
