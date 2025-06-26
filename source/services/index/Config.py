"""
Configuration settings for the Expert AI RAG system.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config(BaseSettings):
    """Configuration settings using Pydantic BaseSettings."""
    
    # Elasticsearch settings
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "localhost")
    ELASTICSEARCH_PORT: int = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    ELASTICSEARCH_INDEX_NAME: str = os.getenv("ELASTICSEARCH_INDEX_NAME", "rag_documents")
    ELASTICSEARCH_USERNAME: Optional[str] = os.getenv("ELASTICSEARCH_USERNAME")
    ELASTICSEARCH_PASSWORD: Optional[str] = os.getenv("ELASTICSEARCH_PASSWORD")
    
    # IBM Watsonx.ai settings
    WATSONX_API_KEY: Optional[str] = os.getenv("WATSONX_API_KEY")
    WATSONX_URL: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    WATSONX_PROJECT_ID: Optional[str] = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_EMBEDDING_MODEL: str = os.getenv("WATSONX_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    


# Global configuration instance
config = Config()