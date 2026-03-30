"""
llms/gemini_llm.py - Gemini wrapper (Pro + Advanced only)
"""

import google.generativeai as genai
from typing import Optional
from collections import deque
import time
import random
from app import config

class GeminiLLM:
    """
    Clean Gemini wrapper:
    - Pro (default reasoning)
    - Advanced (deep reasoning)
    - Rate limiting + retries
    """

    def __init__(self, mode: str = "pro"):
        genai.configure(api_key=config.GEMINI_API_KEY)

        self.mode = mode.lower()
        self.model_name = self._select_model(self.mode)
        self.model = genai.GenerativeModel(self.model_name)

        self.calls = deque(maxlen=config.GEMINI_RATE_LIMIT)
        print(f"🔥 Gemini initialized with model: {self.model_name}")
        self.max_retries = 3
        self.base_delay = 2

    def _select_model(self, mode: str) -> str:
        """Map mode → model"""
        if mode == "advanced":
            return config.GEMINI_ADVANCED_MODEL
        return config.GEMINI_PRO_MODEL

    def _rate_limit_check(self):
        now = time.time()

        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()

        if len(self.calls) >= config.GEMINI_RATE_LIMIT:
            wait_time = 60 - (now - self.calls[0]) + 1
            print(f"⏳ Gemini ({self.mode}) waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            
            now = time.time()
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
        self.calls.append(now)

    def _generate(self, prompt: str) -> Optional[str]:
        response = self.model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return None

    def invoke(self, prompt: str) -> str:
        self._rate_limit_check()

        for attempt in range(self.max_retries):
            try:
                result = self._generate(prompt)

                if result:
                    return result

                raise ValueError("Empty response")

            except Exception as e:
                wait = self.base_delay * (2 ** attempt) + random.uniform(0, 1)

                print(
                    f"⚠️ Gemini ({self.mode}) error: {str(e)} | "
                    f"Retry {attempt+1}/{self.max_retries} in {wait:.1f}s"
                )

                time.sleep(wait)

        return f"❌ Gemini ({self.mode}) failed after retries."

    def invoke_with_context(self, query: str, context: str) -> str:
        prompt = f"""
        You are an AI research assistant.

        Context:
          {context}

        Question:
          {query}
        Instructions:
          - Give structured answer
          - Cite context
          - Avoid hallucination
        """
        return self.invoke(prompt)
if __name__ == "__main__":
    llm = GeminiLLM()
    
    # Test query
    response = llm.invoke("Explain attention mechanism in transformers.")
    print(f"Response: {response}")