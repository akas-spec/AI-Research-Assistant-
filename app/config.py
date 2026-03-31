"""
config.py - Configuration for API keys and model settings
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")


# Groq Models
GROQ_FAST_MODEL = "llama-3.1-8b-instant"        
GROQ_QUALITY_MODEL = "llama-3.3-70b-versatile" 

# Gemini Models for better quality
GEMINI_PRO_MODEL = "models/gemini-2.5-flash"
GEMINI_ADVANCED_MODEL = "models/gemini-2.5-flash"


GROQ_RATE_LIMIT = 30
GEMINI_RATE_LIMIT = 60


DATA_RAW_PATH = "data/raw/"
CHROMA_DB_PATH = "data/chroma_db/"


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

RETRIEVAL_TOP_K = 5

os.makedirs(DATA_RAW_PATH, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
