"""
llms/groq_llm.py - Groq API wrapper (FAST inference)
"""

from groq import Groq
from typing import List, Dict
from app import config
from collections import deque
import time

class GroqLLM:
    """
    Wrapper for Groq API with rate limiting
    """
    
    def __init__(self):
        
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.calls = deque(maxlen=config.GROQ_RATE_LIMIT)
    
    def _rate_limit_check(self):
        """Enforce rate limiting"""
        now = time.time()
        
        # Remove calls older than 60 seconds
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        
        # Wait if at limit
        if len(self.calls) >= config.GROQ_RATE_LIMIT:
            wait_time = 60 - (now - self.calls[0]) + 1
            print(f"⏳ Groq rate limit reached. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

            while self.calls and now - self.calls[0] > 60:
              self.calls.popleft()
        
        self.calls.append(now)
    
    def invoke(
        self, 
        prompt: str, 
        model: str = config.GROQ_QUALITY_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        system_message: str = None
    ) -> str:
        """
        Call Groq API
        
        Args:
            prompt: User prompt
            model: Model name (fast or quality)
            temperature: Creativity (0-1)
            max_tokens: Max response length
            system_message: Optional system prompt
        
        Returns:
            Model response text
        """
        self._rate_limit_check()
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def classify(self, prompt: str) -> str:
        """
        Fast classification using small model
        """
        return self.invoke(
            prompt=prompt,
            model=config.GROQ_FAST_MODEL,
            temperature=0.1,
            max_tokens=100
        )


# Example usage
if __name__ == "__main__":
    llm = GroqLLM()
    
    # Test query
    response = llm.invoke("Explain transformers in one sentence.")
    print(f"Response: {response}")