"""
Elasticsearch indexer for RAG system.
Processes PDF documents, extracts text, splits into chunks, generates embeddings, and indexes to Elasticsearch.
"""

import os
import io
import tempfile
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from elasticsearch import Elasticsearch
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()


class Elasticsearch_Index:
    """
    Elasticsearch indexer class for RAG system.
    Handles PDF processing, text splitting, embedding generation, and Elasticsearch indexing.
    """
    
    def __init__(self):
        """Initialize the Elasticsearch indexer with credentials from environment variables."""
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
        
        # Initialize embedding model
        self.embedding_model = self._create_embedding_model()
        
        print("Elasticsearch_Index initialized successfully!")
    
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
    
    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """
        Convert a PDF document to markdown format using docling.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Markdown content of the PDF
        """
        converter = DocumentConverter()
        doc = converter.convert(pdf_path).document
        return doc.export_to_markdown()
    
    def convert_pdf_bytes_to_markdown(self, pdf_bytes: bytes) -> str:
        """
        Convert PDF bytes to markdown format using docling.
        
        Args:
            pdf_bytes (bytes): PDF file content as bytes
            
        Returns:
            str: Markdown content of the PDF
        """
        # Create a temporary file from bytes since docling requires a file path
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name
        
        try:
            converter = DocumentConverter()
            doc = converter.convert(temp_file_path).document
            return doc.export_to_markdown()
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def split_markdown_by_headers_and_chunks(self, markdown_content: str, chunk_size: int = 250, chunk_overlap: int = 30):
        """
        Split markdown content first by headers (H1 and H2), then by chunk size.
        
        Args:
            markdown_content (str): The markdown content to split
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Returns:
            list: List of document chunks
        """
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        
        # First split by markdown headers
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(markdown_content)
        
        # Then split by character count
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Split the header-based chunks further by character count
        splits = text_splitter.split_documents(md_header_splits)
        
        return splits
    
    def generate_embeddings_for_chunks(self, chunks):
        """
        Generate embeddings for document chunks using Watsonx.ai.
        
        Args:
            chunks (list): List of document chunks from langchain
            
        Returns:
            list: List of dictionaries containing chunk content, metadata, and embeddings
        """
        # Extract text content from chunks
        texts = [chunk.page_content for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings
        embedding_vectors = self.embedding_model.embed_documents(texts=texts)
        
        # Combine chunks with their embeddings
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            # Handle the embedding response structure
            embedding_vector = []
            if isinstance(embedding_vectors, list) and i < len(embedding_vectors):
                embedding_vector = embedding_vectors[i]
            elif hasattr(embedding_vectors, 'get') and embedding_vectors.get('results'):
                results = embedding_vectors.get('results', [])
                if i < len(results):
                    embedding_vector = results[i].get('embedding', [])
            
            enriched_chunk = {
                'content': chunk.page_content,
                'metadata': chunk.metadata,
                'embedding': embedding_vector
            }
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def index_chunks_to_elasticsearch(self, enriched_chunks, document_name: str, index_name: str = None):
        """
        Index enriched chunks with embeddings to Elasticsearch.
        
        Args:
            enriched_chunks (list): List of dictionaries containing chunk content, metadata, and embeddings
            document_name (str): Name of the source document
            index_name (str): Elasticsearch index name (optional, uses default if not provided)
            
        Returns:
            dict: Indexing results and statistics
        """
        if index_name is None:
            index_name = self.es_default_index
        
        print(f"Indexing {len(enriched_chunks)} chunks to index '{index_name}'")
        
        # Index each chunk
        indexed_count = 0
        failed_count = 0
        results = []
        
        for i, chunk in enumerate(enriched_chunks):
            try:
                # Create document structure for Elasticsearch
                doc = {
                    'content': chunk['content'],
                    'embedding': chunk['embedding'],
                    'metadata': chunk['metadata'],
                    'document_name': document_name,
                    'chunk_id': i,
                    'content_length': len(chunk['content']),
                    'embedding_dimensions': len(chunk['embedding']),
                    'indexed_at': datetime.now().isoformat()
                }
                
                # Generate unique document ID
                doc_id = f"{document_name.replace(' ', '_').lower()}_{i}_{str(uuid.uuid4())[:8]}"
                
                # Index the document
                response = self.es_client.index(
                    index=index_name,
                    id=doc_id,
                    body=doc
                )
                
                results.append({
                    'chunk_id': i,
                    'doc_id': doc_id,
                    'status': 'success',
                    'response': response['result']
                })
                indexed_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Indexed {i + 1}/{len(enriched_chunks)} chunks...")
                    
            except Exception as e:
                print(f"Failed to index chunk {i}: {e}")
                results.append({
                    'chunk_id': i,
                    'status': 'failed',
                    'error': str(e)
                })
                failed_count += 1
        
        # Return indexing statistics
        return {
            'total_chunks': len(enriched_chunks),
            'indexed_successfully': indexed_count,
            'failed_to_index': failed_count,
            'index_name': index_name,
            'results': results
        }
    
    def index_pdf_file(self, pdf_bytes: bytes, document_name: str, index_name: str = None, 
                       chunk_size: int = 500, chunk_overlap: int = 50):
        """
        General method to index a PDF file from bytes to Elasticsearch.
        
        Args:
            pdf_bytes (bytes): PDF file content as bytes
            document_name (str): Name of the document for identification
            index_name (str): Elasticsearch index name (optional, uses default if not provided)
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Returns:
            dict: Complete indexing results and statistics
        """
        print(f"Starting indexing process for document: {document_name}")
        
        try:
            # Step 1: Convert PDF bytes to markdown
            print("Converting PDF to markdown...")
            markdown_content = self.convert_pdf_bytes_to_markdown(pdf_bytes)
            print(f"Conversion successful! Total markdown length: {len(markdown_content)} characters")
            
            # Step 2: Split markdown into chunks
            print("Splitting markdown into chunks...")
            chunks = self.split_markdown_by_headers_and_chunks(markdown_content, chunk_size, chunk_overlap)
            print(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            print("Generating embeddings...")
            enriched_chunks = self.generate_embeddings_for_chunks(chunks)
            print(f"Generated embeddings for {len(enriched_chunks)} chunks")
            
            # Step 4: Index to Elasticsearch
            print("Indexing to Elasticsearch...")
            indexing_results = self.index_chunks_to_elasticsearch(enriched_chunks, document_name, index_name)
            
            print(f"Indexing completed successfully!")
            print(f"Total chunks: {indexing_results['total_chunks']}")
            print(f"Successfully indexed: {indexing_results['indexed_successfully']}")
            print(f"Failed to index: {indexing_results['failed_to_index']}")
            
            return {
                'status': 'success',
                'document_name': document_name,
                'markdown_length': len(markdown_content),
                'chunks_created': len(chunks),
                'indexing_results': indexing_results
            }
            
        except Exception as e:
            error_msg = f"Error processing document {document_name}: {e}"
            print(error_msg)
            return {
                'status': 'error',
                'document_name': document_name,
                'error': error_msg
            }
    
    def convert_attention_pdf_to_markdown(self) -> str:
        """
        Convert the 'Attention Is All You Need.pdf' to markdown.
        
        Returns:
            str: Markdown content of the PDF
        """
        # Get the path to the PDF file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, "..", "..", "Attention Is All You Need.pdf")
        
        return self.convert_pdf_to_markdown(pdf_path)


if __name__ == "__main__":
    print("Testing Elasticsearch_Index class...")
    
    try:
        # Initialize the indexer
        indexer = Elasticsearch_Index()
        
        print("\n" + "="*60)
        print("TEST 1: Converting 'Attention Is All You Need.pdf' to markdown")
        print("="*60)
        
        markdown_content = indexer.convert_attention_pdf_to_markdown()
        print("Conversion successful!")
        print(f"Total markdown length: {len(markdown_content)} characters")
        
        print("\n" + "="*60)
        print("TEST 2: Testing complete indexing pipeline with class methods")
        print("="*60)
        
        # Split the markdown content
        chunks = indexer.split_markdown_by_headers_and_chunks(markdown_content, chunk_size=500, chunk_overlap=50)
        print(f"Total chunks created: {len(chunks)}")
        
        # Generate embeddings for first 3 chunks as a test
        test_chunks = chunks[:3]
        enriched_chunks = indexer.generate_embeddings_for_chunks(test_chunks)
        print(f"Generated embeddings for {len(enriched_chunks)} chunks")
        
        # Index to Elasticsearch
        indexing_results = indexer.index_chunks_to_elasticsearch(
            enriched_chunks, 
            "Attention Is All You Need - Class Test",
            "rag_documents"
        )
        
        print(f"\nIndexing Results:")
        print(f"Total chunks: {indexing_results['total_chunks']}")
        print(f"Successfully indexed: {indexing_results['indexed_successfully']}")
        print(f"Failed to index: {indexing_results['failed_to_index']}")
        
        print("\n" + "="*60)
        print("TEST 3: Testing general PDF indexing method with bytes")
        print("="*60)
        
        # Read the PDF file as bytes for testing the general method
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, "..", "..", "Attention Is All You Need.pdf")
        
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        print(f"Loaded PDF file: {len(pdf_bytes)} bytes")
        
        # Test the general indexing method
        result = indexer.index_pdf_file(
            pdf_bytes=pdf_bytes,
            document_name="Attention Is All You Need - Bytes Test",
            index_name="rag_documents_test",
            chunk_size=600,
            chunk_overlap=60
        )
        
        print(f"\nGeneral indexing method result:")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Document: {result['document_name']}")
            print(f"Markdown length: {result['markdown_length']} characters")
            print(f"Chunks created: {result['chunks_created']}")
            print(f"Successfully indexed: {result['indexing_results']['indexed_successfully']}")
        else:
            print(f"Error: {result['error']}")
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

