"""
Elasticsearch indexer for RAG system.
Processes PDF documents, extracts text, splits into chunks, generates embeddings, and indexes to Elasticsearch.
"""

import os
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


def convert_pdf_to_markdown(pdf_path: str) -> str:
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


def convert_attention_pdf_to_markdown() -> str:
    """
    Convert the 'Attention Is All You Need.pdf' to markdown.
    
    Returns:
        str: Markdown content of the PDF
    """
    # Get the path to the PDF file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "..", "..", "Attention Is All You Need.pdf")
    
    return convert_pdf_to_markdown(pdf_path)


def split_markdown_by_headers_and_chunks(markdown_content: str, chunk_size: int = 250, chunk_overlap: int = 30):
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


def generate_embeddings_for_chunks(chunks):
    """
    Generate embeddings for document chunks using Watsonx.ai.
    
    Args:
        chunks (list): List of document chunks from langchain
        
    Returns:
        list: List of dictionaries containing chunk content, metadata, and embeddings
    """
    # Get credentials from environment variables
    api_key = os.getenv('WATSONX_API_KEY')
    project_id = os.getenv('WATSONX_PROJECT_ID')
    url = os.getenv('WATSONX_URL')
    
    if not all([api_key, project_id, url]):
        raise ValueError("Missing Watsonx.ai credentials in environment variables")
    
    # Configure embedding parameters
    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
        EmbedParams.RETURN_OPTIONS: {
            'input_text': True
        }
    }
    
    # Initialize embedding model
    embedding = Embeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
        params=embed_params,
        credentials=Credentials(
            api_key=api_key.strip("'\""),  # Remove quotes if present
            url=url.strip("'\"")
        ),
        project_id=project_id.strip("'\"")
    )
    
    # Extract text content from chunks
    texts = [chunk.page_content for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    
    # Generate embeddings
    embedding_vectors = embedding.embed_documents(texts=texts)
    
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


def index_chunks_to_elasticsearch(enriched_chunks, document_name="Attention Is All You Need"):
    """
    Index enriched chunks with embeddings to Elasticsearch.
    
    Args:
        enriched_chunks (list): List of dictionaries containing chunk content, metadata, and embeddings
        document_name (str): Name of the source document
        
    Returns:
        dict: Indexing results and statistics
    """
    # Get Elasticsearch credentials from environment variables
    es_host = os.getenv('ELASTICSEARCH_HOST')
    es_port = os.getenv('ELASTICSEARCH_PORT')
    es_username = os.getenv('ELASTICSEARCH_USERNAME')
    es_password = os.getenv('ELASTICSEARCH_PASSWORD')
    es_index = os.getenv('ELASTICSEARCH_INDEX_NAME')
    
    if not all([es_host, es_port, es_username, es_password, es_index]):
        raise ValueError("Missing Elasticsearch credentials in environment variables")
    
    # Create Elasticsearch client - using simple URL format that works with IBM Cloud
    es_url = f"https://{es_username}:{es_password}@{es_host}:{es_port}"
    
    es_client = Elasticsearch(
        hosts=[es_url],
        verify_certs=False,  # Disable SSL certificate verification for self-signed cert
        ssl_show_warn=False,  # Disable SSL warnings
        request_timeout=60,
        headers={"Accept": "application/json", "Content-Type": "application/json"}  # Force standard JSON headers
    )
    
    print(f"Connecting to Elasticsearch at {es_host}:{es_port}")
    
    # Check if Elasticsearch is available by trying to get cluster info directly
    try:
        # Try to get cluster info directly instead of ping
        cluster_info = es_client.info()
        print(f"Connected successfully! Cluster: {cluster_info.get('cluster_name', 'Unknown')}")
        print(f"Elasticsearch version: {cluster_info.get('version', {}).get('number', 'Unknown')}")
        
    except Exception as e:
        print(f"Connection error details: {e}")
        raise ConnectionError(f"Could not connect to Elasticsearch: {e}")
    
    print(f"Connected successfully! Indexing {len(enriched_chunks)} chunks to index '{es_index}'")
    
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
            response = es_client.index(
                index=es_index,
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
        'index_name': es_index,
        'results': results
    }


if __name__ == "__main__":
    print("Converting 'Attention Is All You Need.pdf' to markdown...")
    try:
        markdown_content = convert_attention_pdf_to_markdown()
        print("Conversion successful!")
        print(f"Total markdown length: {len(markdown_content)} characters")
        
        print("\n" + "="*50)
        print("SPLITTING MARKDOWN INTO CHUNKS:")
        print("="*50)
        
        # Split the markdown content
        chunks = split_markdown_by_headers_and_chunks(markdown_content, chunk_size=500, chunk_overlap=50)
        
        print(f"Total chunks created: {len(chunks)}")
        print("\nFirst 3 chunks:")
        print("-" * 30)
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Content length: {len(chunk.page_content)} characters")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content preview: {chunk.page_content[:200]}...")
            print("-" * 30)
        
        print("\n" + "="*50)
        print("GENERATING EMBEDDINGS:")
        print("="*50)
        
        # Generate embeddings for first 5 chunks as a test
        test_chunks = chunks[:5]
        enriched_chunks = generate_embeddings_for_chunks(test_chunks)
        
        print(f"Generated embeddings for {len(enriched_chunks)} chunks")
        
        for i, enriched_chunk in enumerate(enriched_chunks):
            print(f"\nEnriched Chunk {i+1}:")
            print(f"Content length: {len(enriched_chunk['content'])} characters")
            print(f"Metadata: {enriched_chunk['metadata']}")
            print(f"Embedding dimensions: {len(enriched_chunk['embedding'])}")
            print(f"Embedding preview: {enriched_chunk['embedding'][:5]}...")
            print("-" * 30)
        
        print("\n" + "="*50)
        print("INDEXING TO ELASTICSEARCH:")
        print("="*50)
        
        # Index the enriched chunks to Elasticsearch
        indexing_results = index_chunks_to_elasticsearch(enriched_chunks)
        
        print(f"\nIndexing Results:")
        print(f"Total chunks: {indexing_results['total_chunks']}")
        print(f"Successfully indexed: {indexing_results['indexed_successfully']}")
        print(f"Failed to index: {indexing_results['failed_to_index']}")
        print(f"Index name: {indexing_results['index_name']}")
        
        # Show some successful indexing results
        successful_results = [r for r in indexing_results['results'] if r['status'] == 'success']
        if successful_results:
            print(f"\nFirst few successful indexing operations:")
            for result in successful_results[:3]:
                print(f"- Chunk {result['chunk_id']}: {result['response']} (ID: {result['doc_id']})")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

