"""
Document Service for ExpertORT Agent.
Handles document indexing and retrieval operations.
"""

import sys
import os
from typing import List, Dict, Any, Optional

# Add the parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))

from services.index.elasticsearch_index import Elasticsearch_Index
from services.retrieval.elasticsearch_retrieval import Elasticsearch_Retrieval


class DocumentService:
    """
    Service class for document operations.
    Handles file indexing and query retrieval through Elasticsearch.
    """
    
    def __init__(self):
        """Initialize the document service with indexing and retrieval capabilities."""
        print("🗂️ Initializing Document Service...")
        
        # Initialize indexing system
        try:
            self.indexer = Elasticsearch_Index()
            print("✅ Indexer initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing indexer: {e}")
            self.indexer = None
        
        # Initialize retrieval system
        try:
            self.retriever = Elasticsearch_Retrieval()
            print("✅ Retriever initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing retriever: {e}")
            self.retriever = None
        
        print("✅ Document Service initialized!")
    
    def index_file(self, file_bytes: bytes, filename: str, index_name: str, 
                   chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
        """
        Index a PDF file to Elasticsearch.
        
        Args:
            file_bytes (bytes): PDF file content as bytes
            filename (str): Name of the file
            index_name (str): Elasticsearch index name
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Returns:
            Dict[str, Any]: Indexing results
        """
        if not self.indexer:
            return {
                'status': 'error',
                'message': 'Indexer not available. Check Elasticsearch and Watsonx.ai configuration.',
                'document_name': filename
            }
        
        try:
            print(f"📄 Starting indexing process for file: {filename}")
            
            # Use the indexer to process the file
            result = self.indexer.index_pdf_file(
                pdf_bytes=file_bytes,
                document_name=filename,
                index_name=index_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'message': f'File {filename} indexed successfully',
                    'document_name': filename,
                    'total_chunks': result['chunks_created'],
                    'indexed_successfully': result['indexing_results']['indexed_successfully'],
                    'failed_to_index': result['indexing_results']['failed_to_index'],
                    'index_name': index_name
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Error indexing file {filename}: {result["error"]}',
                    'document_name': filename
                }
                
        except Exception as e:
            error_message = f"Error processing file {filename}: {str(e)}"
            print(f"❌ {error_message}")
            return {
                'status': 'error',
                'message': error_message,
                'document_name': filename
            }
    
    def query_documents(self, query: str, k: int = 10, top_p: int = 3, 
                       min_score: float = 0.1, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Query documents using semantic search and reranking.
        
        Args:
            query (str): Search query
            k (int): Number of initial candidates to retrieve
            top_p (int): Number of final results after reranking
            min_score (float): Minimum similarity score threshold
            index_name (Optional[str]): Elasticsearch index name
            
        Returns:
            Dict[str, Any]: Query results
        """
        if not self.retriever:
            return {
                'status': 'error',
                'message': 'Retriever not available. Check Elasticsearch and Watsonx.ai configuration.',
                'query': query,
                'total_candidates': 0,
                'returned_results': 0,
                'results': [],
                'search_metadata': {}
            }
        
        try:
            print(f"🔍 Processing query: '{query}'")
            
            # Step 1: Semantic search to get initial candidates
            initial_results = self.retriever.retrieve_top_k_documents(
                query=query,
                k=k,
                index_name=index_name,
                min_score=min_score
            )
            
            if not initial_results:
                return {
                    'status': 'success',
                    'message': f'No relevant documents found for query: {query}',
                    'query': query,
                    'total_candidates': 0,
                    'returned_results': 0,
                    'results': [],
                    'search_metadata': {
                        'semantic_search_candidates': 0,
                        'reranking_applied': False,
                        'min_score_threshold': min_score
                    }
                }
            
            print(f"📄 Found {len(initial_results)} initial candidates")
            
            # Step 2: Apply reranking to get the most relevant results
            reranked_results = self.retriever.rerank_documents(
                query=query,
                documents=initial_results,
                top_p=top_p
            )
            
            print(f"🎯 Reranking completed. Returning {len(reranked_results)} results")
            
            # Step 3: Format results
            formatted_results = []
            for result in reranked_results:
                formatted_result = {
                    'id': result.get('id', ''),
                    'score': result.get('score', 0.0),
                    'rerank_score': result.get('rerank_score'),
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'document_name': result.get('document_name', ''),
                    'chunk_id': result.get('chunk_id', 0),
                    'content_length': result.get('content_length', 0),
                    'indexed_at': result.get('indexed_at', '')
                }
                formatted_results.append(formatted_result)
            
            return {
                'status': 'success',
                'message': f'Found {len(formatted_results)} relevant documents',
                'query': query,
                'total_candidates': len(initial_results),
                'returned_results': len(formatted_results),
                'results': formatted_results,
                'search_metadata': {
                    'semantic_search_candidates': len(initial_results),
                    'reranking_applied': True,
                    'min_score_threshold': min_score,
                    'chunk_size_requested': k,
                    'final_results_requested': top_p
                }
            }
            
        except Exception as e:
            error_message = f"Error processing query '{query}': {str(e)}"
            print(f"❌ {error_message}")
            return {
                'status': 'error',
                'message': error_message,
                'query': query,
                'total_candidates': 0,
                'returned_results': 0,
                'results': [],
                'search_metadata': {}
            }
    
    def get_index_statistics(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about an index.
        
        Args:
            index_name (Optional[str]): Elasticsearch index name
            
        Returns:
            Dict[str, Any]: Index statistics
        """
        if not self.retriever:
            return {'error': 'Retriever not available'}
        
        try:
            return self.retriever.get_index_statistics(index_name)
        except Exception as e:
            return {'error': str(e)}
    
    def search_by_document_name(self, document_name: str, k: int = 10, 
                               index_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for documents by name.
        
        Args:
            document_name (str): Name of the document to search for
            k (int): Number of results to return
            index_name (Optional[str]): Elasticsearch index name
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        if not self.retriever:
            return []
        
        try:
            return self.retriever.search_by_document_name(document_name, k, index_name)
        except Exception as e:
            print(f"Error searching by document name: {e}")
            return []
