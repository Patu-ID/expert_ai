"""
Shared service instances for the ExpertORT Agent.
Implements lazy initialization and singleton pattern to avoid multiple initialization of expensive resources.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SharedServices:
    """
    Singleton class that manages shared service instances.
    Implements lazy initialization to create instances only when needed.
    """
    
    _instance = None
    _elasticsearch_index = None
    _elasticsearch_retrieval = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedServices, cls).__new__(cls)
        return cls._instance
    
    def get_elasticsearch_index(self):
        """
        Get or create the Elasticsearch Index instance.
        Uses lazy initialization to create only when needed.
        """
        if self._elasticsearch_index is None:
            print("🗂️ Initializing shared Elasticsearch Index...")
            from services.index.elasticsearch_index import Elasticsearch_Index
            self._elasticsearch_index = Elasticsearch_Index()
            print("✅ Shared Elasticsearch Index initialized!")
        return self._elasticsearch_index
    
    def get_elasticsearch_retrieval(self):
        """
        Get or create the Elasticsearch Retrieval instance.
        Uses lazy initialization to create only when needed.
        """
        if self._elasticsearch_retrieval is None:
            print("🔍 Initializing shared Elasticsearch Retrieval...")
            from services.retrieval.elasticsearch_retrieval import Elasticsearch_Retrieval
            self._elasticsearch_retrieval = Elasticsearch_Retrieval()
            print("✅ Shared Elasticsearch Retrieval initialized!")
        return self._elasticsearch_retrieval
    
    def clear_instances(self):
        """
        Clear all cached instances.
        Useful for testing or when services need to be reinitialized.
        """
        self._elasticsearch_index = None
        self._elasticsearch_retrieval = None
        print("🔄 Shared service instances cleared")


# Global singleton instance
shared_services = SharedServices()
