from dotenv import load_dotenv
from pathlib import Path
import os

# Explicitly locate .env relative to this file, so it works
# no matter what directory you run scripts/pytest from.
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KE")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Optional: fail loudly early if something critical is missing,
# instead of failing later inside some library deep in a traceback.
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Check your .env file.")
