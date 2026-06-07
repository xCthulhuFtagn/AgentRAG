"""Configuration — DeepSeek API, FastEmbed, LanceDB settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# DeepSeek API (OpenAI-compatible)
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# LanceDB
LANCE_DB_PATH: str = os.getenv("LANCE_DB_PATH", "./lancedb_data")

# Agent loop
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))

# Embeddings (FastEmbed)
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

# Project root
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
