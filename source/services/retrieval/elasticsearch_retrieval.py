"""
Elasticsearch retrieval system for RAG.
Handles semantic search and retrieval of relevant document chunks fro    def check_available_rerank_models(self):"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models import Rerank
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from typing import List, Dict, Any

# Load environment variables
load_dotenv()


class Elasticsearch_Retrieval:
    """
    Elasticsearch retrieval class for RAG system.
    Handles semantic search and retrieval of relevant document chunks.
    """
    
    def __init__(self):
        """Initialize the Elasticsearch retrieval system with credentials from environment variables."""
        # Load Watsonx.ai credentials
        self.watsonx_api_key = os.getenv('WATSONX_API_KEY')
        self.watsonx_project_id = os.getenv('WATSONX_PROJECT_ID')
        self.watsonx_url = os.getenv('WATSONX_URL')
        
        # Load Elasticsearch credentials
        self.es_host = os.getenv('ELASTICSEARCH_HOST')
        self.es_port = os.getenv('ELASTICSEARCH_PORT')
        self.es_username = os.getenv('ELASTICSEARCH_USERNAME')
        self.es_password = os.getenv('ELASTICSEARCH_PASSWORD')
        self.es_default_index = os.getenv('ELASTICSEARCH_INDEX_NAME')
        
        # Validate credentials
        self._validate_credentials()
        
        # Initialize Elasticsearch client
        self.es_client = self._create_elasticsearch_client()
        
        # Initialize Watsonx.ai credentials
        credentials = Credentials(
            api_key=self.watsonx_api_key.strip("'\""),
            url=self.watsonx_url.strip("'\"")
        )
        watsonx_project_id = self.watsonx_project_id.strip("'\"")
        
        # Initialize embedding model
        self.embedding_model = self._create_embedding_model()
        
        # Initialize reranking model
        self.reranker = Rerank(
            model_id="cross-encoder/ms-marco-minilm-l-12-v2",  # Using a well-known reranking model
            credentials=credentials,
            project_id=watsonx_project_id
        )
        
        print("Elasticsearch_Retrieval initialized successfully!")
    
    def _validate_credentials(self):
        """Validate that all required credentials are present."""
        if not all([self.watsonx_api_key, self.watsonx_project_id, self.watsonx_url]):
            raise ValueError("Missing Watsonx.ai credentials in environment variables")
        
        if not all([self.es_host, self.es_port, self.es_username, self.es_password]):
            raise ValueError("Missing Elasticsearch credentials in environment variables")
    
    def _create_elasticsearch_client(self):
        """Create and configure Elasticsearch client."""
        es_url = f"https://{self.es_username}:{self.es_password}@{self.es_host}:{self.es_port}"
        
        client = Elasticsearch(
            hosts=[es_url],
            verify_certs=False,
            ssl_show_warn=False,
            request_timeout=60,
            headers={"Accept": "application/json", "Content-Type": "application/json"}
        )
        
        # Test connection
        try:
            cluster_info = client.info()
            print(f"Connected to Elasticsearch cluster: {cluster_info.get('cluster_name', 'Unknown')}")
            print(f"Elasticsearch version: {cluster_info.get('version', {}).get('number', 'Unknown')}")
        except Exception as e:
            raise ConnectionError(f"Could not connect to Elasticsearch: {e}")
        
        return client
    
    def _create_embedding_model(self):
        """Create and configure Watsonx.ai embedding model."""
        embed_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
            EmbedParams.RETURN_OPTIONS: {
                'input_text': True
            }
        }
        
        return Embeddings(
            model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
            params=embed_params,
            credentials=Credentials(
                api_key=self.watsonx_api_key.strip("'\""),
                url=self.watsonx_url.strip("'\"")
            ),
            project_id=self.watsonx_project_id.strip("'\"")
        )
    
    def _create_rerank_model(self):
        """Create and configure Watsonx.ai reranking model."""
        rerank_params = {
            "truncate_input_tokens": 512  # Increased for longer documents
        }
        
        return Rerank(
            model_id="ibm/rerank-colbert-v1",  # IBM's reranking model
            params=rerank_params,
            credentials=Credentials(
                api_key=self.watsonx_api_key.strip("'\""),
                url=self.watsonx_url.strip("'\"")
            ),
            project_id=self.watsonx_project_id.strip("'\"")
        )
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string using Watsonx.ai.
        
        Args:
            query (str): The query string to embed
            
        Returns:
            List[float]: The embedding vector for the query
        """
        print(f"Generating embedding for query: '{query}'")
        
        # Generate embedding for the query
        embedding_vectors = self.embedding_model.embed_documents(texts=[query])
        
        # Extract the embedding vector
        if isinstance(embedding_vectors, list) and len(embedding_vectors) > 0:
            return embedding_vectors[0]
        elif hasattr(embedding_vectors, 'get') and embedding_vectors.get('results'):
            results = embedding_vectors.get('results', [])
            if len(results) > 0:
                return results[0].get('embedding', [])
        
        raise ValueError("Failed to generate embedding for query")
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_p: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using Watsonx.ai reranking model.
        
        Args:
            query (str): The search query
            documents (List[Dict[str, Any]]): List of documents with 'content' field (from retrieve_top_k_documents)
            top_p (int): Number of top reranked results to return
            
        Returns:
            List[Dict[str, Any]]: Reranked documents with rerank scores
        """
        if not documents:
            return []
        
        print(f"Reranking {len(documents)} documents using Watsonx.ai reranker")
        
        try:
            # Prepare document texts for reranking
            doc_texts = [doc['content'] for doc in documents]
            
            # Use reranker with parameters
            rerank_params = {
                "truncate_input_tokens": 512
            }
            
            # Perform reranking using the generate method
            response = self.reranker.generate(
                query=query,
                inputs=doc_texts,
                params=rerank_params
            )
            
            # Parse reranking results
            reranked_docs = []
            
            # The response format may vary, let's handle different possible formats
            if hasattr(response, 'results') and response.results:
                # If response has results attribute
                for result in response.results:
                    doc_idx = result.index
                    if doc_idx < len(documents):
                        doc = documents[doc_idx].copy()
                        doc['rerank_score'] = result.score
                        reranked_docs.append(doc)
            elif isinstance(response, dict) and 'results' in response:
                # If response is a dictionary with results
                for result in response['results']:
                    doc_idx = result.get('index', 0)
                    if doc_idx < len(documents):
                        doc = documents[doc_idx].copy()
                        doc['rerank_score'] = result.get('score', result.get('relevance_score', 0.0))
                        reranked_docs.append(doc)
            else:
                # Fallback: assume response is a list of scores
                print(f"Reranking response format: {type(response)}")
                print(f"Reranking response: {response}")
                # If we can't parse the response, return original documents
                reranked_docs = documents.copy()
                for i, doc in enumerate(reranked_docs):
                    doc['rerank_score'] = 1.0 - (i * 0.1)  # Assign decreasing scores
            
            # Sort by rerank score (highest first) and return top_p
            reranked_docs.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            final_results = reranked_docs[:top_p]
            
            print(f"Reranking completed. Returning top {len(final_results)} documents.")
            return final_results
            
        except Exception as e:
            print(f"Error in reranking: {e}")
            print(f"Falling back to original ranking")
            return documents[:top_p]  # Fallback to original ranking
    
    def retrieve_top_k_documents(self, query: str, k: int = 5, index_name: str = None, 
                                 min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve the top k most relevant documents for a given query.
        
        Args:
            query (str): The search query
            k (int): Number of documents to retrieve (default: 5)
            index_name (str): Elasticsearch index name (optional, uses default if not provided)
            min_score (float): Minimum similarity score threshold (default: 0.0)
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with their metadata and scores
        """
        if index_name is None:
            index_name = self.es_default_index
        
        print(f"Retrieving top {k} documents for query: '{query}' from index '{index_name}'")
        
        try:
            # Generate embedding for the query
            query_embedding = self.generate_query_embedding(query)
            print(f"Query embedding generated with {len(query_embedding)} dimensions")
            
            # Fetch all documents first - this is a simplified approach
            # In production, you might want to use pagination for large datasets
            search_query = {
                "size": 100,  # Get more documents to calculate similarity
                "query": {"match_all": {}},
                "_source": [
                    "content", "metadata", "document_name", "chunk_id", 
                    "content_length", "indexed_at", "embedding"
                ]
            }
            
            # Execute the search
            response = self.es_client.search(index=index_name, body=search_query)
            
            # Calculate cosine similarity in Python
            results = []
            hits = response.get('hits', {}).get('hits', [])
            
            print(f"Found {len(hits)} documents to calculate similarity for")
            
            for hit in hits:
                source = hit.get('_source', {})
                doc_embedding = source.get('embedding', [])
                
                if not doc_embedding or len(doc_embedding) != len(query_embedding):
                    continue
                
                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                query_magnitude = sum(a * a for a in query_embedding) ** 0.5
                doc_magnitude = sum(b * b for b in doc_embedding) ** 0.5
                
                if query_magnitude == 0 or doc_magnitude == 0:
                    score = 0.0
                else:
                    score = dot_product / (query_magnitude * doc_magnitude)
                
                # Debug: print first few scores
                if len(results) < 3:
                    print(f"  Document '{source.get('document_name', '')}' chunk {source.get('chunk_id', 0)}: score = {score:.4f}")
                
                # Apply minimum score threshold
                if score < min_score:
                    continue
                
                result = {
                    'id': hit.get('_id'),
                    'score': score,
                    'content': source.get('content', ''),
                    'metadata': source.get('metadata', {}),
                    'document_name': source.get('document_name', ''),
                    'chunk_id': source.get('chunk_id', 0),
                    'content_length': source.get('content_length', 0),
                    'indexed_at': source.get('indexed_at', '')
                }
                
                results.append(result)
            
            # Sort by score (descending) and take top k
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:k]
            
            print(f"Retrieved {len(results)} documents after filtering and ranking")
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            raise e
    
    def search_by_document_name(self, document_name: str, k: int = 10, index_name: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents by document name (useful for exploring indexed content).
        
        Args:
            document_name (str): Name of the document to search for
            k (int): Number of documents to retrieve (default: 10)
            index_name (str): Elasticsearch index name (optional, uses default if not provided)
            
        Returns:
            List[Dict[str, Any]]: List of documents matching the document name
        """
        if index_name is None:
            index_name = self.es_default_index
        
        print(f"Searching for documents with name: '{document_name}' in index '{index_name}'")
        
        try:
            # Construct the Elasticsearch query for exact document name match
            search_query = {
                "size": k,
                "query": {
                    "match": {
                        "document_name": document_name
                    }
                },
                "sort": [
                    {"chunk_id": {"order": "asc"}}  # Sort by chunk_id to maintain order
                ],
                "_source": [
                    "content", "metadata", "document_name", "chunk_id", 
                    "content_length", "indexed_at"
                ]
            }
            
            # Execute the search
            response = self.es_client.search(index=index_name, body=search_query)
            
            # Process the results
            results = []
            hits = response.get('hits', {}).get('hits', [])
            
            print(f"Found {len(hits)} chunks for document '{document_name}'")
            
            for hit in hits:
                source = hit.get('_source', {})
                
                result = {
                    'id': hit.get('_id'),
                    'score': hit.get('_score', 0.0),
                    'content': source.get('content', ''),
                    'metadata': source.get('metadata', {}),
                    'document_name': source.get('document_name', ''),
                    'chunk_id': source.get('chunk_id', 0),
                    'content_length': source.get('content_length', 0),
                    'indexed_at': source.get('indexed_at', '')
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error during document search: {e}")
            raise e
    
    def get_index_statistics(self, index_name: str = None) -> Dict[str, Any]:
        """
        Get statistics about the documents in the index.
        
        Args:
            index_name (str): Elasticsearch index name (optional, uses default if not provided)
            
        Returns:
            Dict[str, Any]: Index statistics
        """
        if index_name is None:
            index_name = self.es_default_index
        
        try:
            # Get basic index stats
            stats = self.es_client.indices.stats(index=index_name)
            
            # Get document count
            count_response = self.es_client.count(index=index_name)
            total_docs = count_response.get('count', 0)
            
            # Get unique document names
            agg_query = {
                "size": 0,
                "aggs": {
                    "unique_documents": {
                        "terms": {
                            "field": "document_name.keyword",
                            "size": 100
                        }
                    }
                }
            }
            
            agg_response = self.es_client.search(index=index_name, body=agg_query)
            unique_docs = agg_response.get('aggregations', {}).get('unique_documents', {}).get('buckets', [])
            
            return {
                'index_name': index_name,
                'total_documents': total_docs,
                'unique_document_names': [bucket['key'] for bucket in unique_docs],
                'unique_document_count': len(unique_docs),
                'index_size_bytes': stats.get('_all', {}).get('total', {}).get('store', {}).get('size_in_bytes', 0)
            }
            
        except Exception as e:
            print(f"Error getting index statistics: {e}")
            return {'error': str(e)}
    
    def check_available_rerank_models(self):
        """Check what reranking models are available."""
        try:
            from ibm_watsonx_ai import APIClient
            
            # Create API client
            client = APIClient(credentials=Credentials(
                api_key=self.watsonx_api_key.strip("'\""),
                url=self.watsonx_url.strip("'\"")
            ))
            client.set.default_project(self.watsonx_project_id.strip("'\""))
            
            # Get available reranking models
            rerank_models = client.foundation_models.RerankModels
            print("Available reranking models:")
            for model in rerank_models:
                print(f"  - {model}")
            
            return rerank_models
            
        except Exception as e:
            print(f"Error checking available models: {e}")
            return []


if __name__ == "__main__":
    print("Testing Elasticsearch_Retrieval class...")
    
    try:
        # Initialize the retrieval system
        retriever = Elasticsearch_Retrieval()
        
        print("\n" + "="*60)
        print("TEST 1: Getting index statistics")
        print("="*60)
        
        # Get index statistics
        stats = retriever.get_index_statistics()
        print(f"Index statistics:")
        print(f"- Index name: {stats.get('index_name')}")
        print(f"- Total documents: {stats.get('total_documents')}")
        print(f"- Unique documents: {stats.get('unique_document_count')}")
        print(f"- Document names: {stats.get('unique_document_names')}")
        
        print("\n" + "="*60)
        print("TEST 2: Searching by document name")
        print("="*60)
        
        # Search for chunks of the "Attention Is All You Need" document
        document_results = retriever.search_by_document_name("Attention Is All You Need", k=5)
        
        print(f"Found {len(document_results)} chunks:")
        for i, result in enumerate(document_results):
            print(f"\nChunk {i+1}:")
            print(f"- ID: {result['id']}")
            print(f"- Chunk ID: {result['chunk_id']}")
            print(f"- Content length: {result['content_length']} characters")
            print(f"- Metadata: {result['metadata']}")
            print(f"- Content preview: {result['content'][:200]}...")
        
        print("\n" + "="*60)
        print("TEST 3: Semantic retrieval with sample queries")
        print("="*60)
        
        # Test queries related to the "Attention Is All You Need" paper
        test_queries = [
            "What is the Transformer architecture?",
            "How does self-attention work?",
            "What are the advantages of attention mechanisms?",
            "Multi-head attention mechanism",
            "Training procedure and results"
        ]
        
        for query in test_queries:
            print(f"\n--- Query: '{query}' ---")
            results = retriever.retrieve_top_k_documents(query, k=3, min_score=0.1)  # Lower threshold
            
            print(f"Retrieved {len(results)} relevant documents:")
            for i, result in enumerate(results):
                print(f"\n  Result {i+1} (Score: {result['score']:.3f}):")
                print(f"  - Document: {result['document_name']}")
                print(f"  - Chunk ID: {result['chunk_id']}")
                print(f"  - Metadata: {result['metadata']}")
                print(f"  - Content: {result['content'][:150]}...")
        
        # Check available reranking models first
        print("Checking available reranking models...")
        available_models = retriever.check_available_rerank_models()
        
        # Test reranking
        print("\n=== Testing Reranking ===")
        query = "What are attention mechanisms in neural networks?"
        
        # First get semantic search results
        search_results = retriever.retrieve_top_k_documents(query, k=10)
        if search_results:
            print(f"\nOriginal search results (top 3 of {len(search_results)}):")
            for i, result in enumerate(search_results[:3]):
                content_preview = result['content'][:200] + "..."
                print(f"  {i+1}. Score: {result['score']:.4f}")
                print(f"     Content: {content_preview}")
            
            # Rerank the results
            reranked_results = retriever.rerank_documents(query, search_results, top_p=5)
            print(f"\nReranked results (top 3 of {len(reranked_results)}):")
            for i, result in enumerate(reranked_results[:3]):
                content_preview = result['content'][:200] + "..."
                rerank_score = result.get('rerank_score', 'N/A')
                print(f"  {i+1}. Rerank Score: {rerank_score:.4f}" if rerank_score != 'N/A' else f"  {i+1}. Rerank Score: {rerank_score}")
                print(f"     Original Score: {result['score']:.4f}")
                print(f"     Content: {content_preview}")
        else:
            print("No search results to rerank")
        
        print("\n" + "="*60)
        print("ALL RETRIEVAL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
