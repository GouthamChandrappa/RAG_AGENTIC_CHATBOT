import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# LLM Configuration
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-large"

# Vector Database Configuration
VECTOR_DB_TYPE = "qdrant"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "rag_docs"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEMANTIC_CHUNKING = True

# Evaluation Configuration
EVAL_METRICS = [
    "contextual_precision",
    "contextual_recall", 
    "contextual_relevancy",
    "answer_relevancy",
    "faithfulness"
]

# Agent Configuration
MAX_ITERATIONS = 5
TASK_TIMEOUT = 60  # seconds

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLE_DOCS_DIR = os.path.join(DATA_DIR, "sample_docs")