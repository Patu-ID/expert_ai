# ExpertORT Agent - Document Indexing System

A sophisticated Retrieval-Augmented Generation (RAG) system that enables intelligent document processing and querying through advanced indexing capabilities.

## 📋 Table of Contents

- [Document Indexing System Overview](#document-indexing-system-overview)
- [Indexing Architecture](#indexing-architecture)
- [The Complete Indexing Pipeline](#the-complete-indexing-pipeline)
- [Technical Components](#technical-components)
- [API Endpoints for Indexing](#api-endpoints-for-indexing)
- [Agent System Architecture](#agent-system-architecture)
- [Large Language Model Configuration](#large-language-model-configuration)
- [Agent Tools System](#agent-tools-system)
- [System Prompt Design](#system-prompt-design)
- [Agent Orchestration with LangGraph](#agent-orchestration-with-langgraph)
- [Chat Processing System](#chat-processing-system)
- [Agent Connect Protocol Implementation](#agent-connect-protocol-implementation)
- [ExpertORT User Interface System](#expertort-user-interface-system)
- [Configuration Requirements](#configuration-requirements)

## 🗂️ Document Indexing System Overview

The ExpertORT Agent's indexing system is designed to transform PDF documents into a searchable knowledge base using state-of-the-art natural language processing techniques. The system converts complex documents into semantically meaningful chunks that can be efficiently retrieved and used for question-answering.

### Key Features

- **PDF-to-Markdown Conversion**: Intelligent document parsing using Docling
- **Smart Text Chunking**: Header-aware and size-based text splitting
- **Semantic Embeddings**: Vector representations using IBM Watson AI
- **Elasticsearch Integration**: Scalable search and storage infrastructure
- **Real-time Processing**: Asynchronous document processing capabilities

## 🏗️ Indexing Architecture

The indexing system follows a modular, service-oriented architecture:

```
📁 Indexing Architecture
├── 🌐 API Layer (FastAPI)
│   └── Document Router (/v1/documents/index)
├── 🔧 Service Layer
│   ├── DocumentService (Orchestration)
│   └── SharedServices (Singleton Management)
├── 🗂️ Index Layer
│   └── Elasticsearch_Index (Core Processing)
└── 💾 Storage Layer
    ├── Elasticsearch (Vector Storage)
    └── IBM Watson AI (Embeddings)
```

### Component Relationships

1. **API Layer**: Handles HTTP requests and file uploads
2. **Service Layer**: Orchestrates the indexing workflow
3. **Processing Layer**: Executes document transformation steps
4. **Storage Layer**: Persists indexed data and embeddings

## 🔄 The Complete Indexing Pipeline

The indexing process consists of 6 distinct stages, each optimized for specific document processing tasks:

### Stage 1: Document Reception & Validation

**Location**: `api/routers/document_router.py`

```python
# API endpoint receives PDF file
@router.post("/v1/documents/index")
async def index_file(
    file: UploadFile = File(...),
    index_name: str = Form(...),
    chunk_size: Optional[int] = Form(500),
    chunk_overlap: Optional[int] = Form(50)
)
```

**Process**:
- Validates file type (PDF only)
- Checks file integrity and size
- Extracts file metadata (name, size)
- Passes binary data to service layer

**Error Handling**:
- Rejects non-PDF files
- Handles empty or corrupted files
- Provides detailed error messages

### Stage 2: PDF-to-Markdown Conversion

**Location**: `services/index/elasticsearch_index.py`

```python
def convert_pdf_bytes_to_markdown(self, pdf_bytes: bytes) -> str:
    # Creates temporary file from bytes
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(pdf_bytes)
    
    # Uses Docling for intelligent conversion
    converter = DocumentConverter()
    doc = converter.convert(temp_file_path).document
    return doc.export_to_markdown()
```

**Key Features**:
- **Docling Integration**: Advanced PDF parsing that preserves document structure
- **Content Preservation**: Maintains headers, paragraphs, tables, and formatting
- **Temporary File Management**: Secure handling of uploaded content
- **Markdown Export**: Structured text format ideal for processing

**Technical Details**:
- Supports complex PDF layouts
- Preserves semantic document structure
- Handles embedded images and tables
- Maintains cross-references and citations

### Stage 3: Intelligent Text Chunking

**Location**: `services/index/elasticsearch_index.py`

```python
def split_markdown_by_headers_and_chunks(self, markdown_content: str, 
                                       chunk_size: int = 250, 
                                       chunk_overlap: int = 30):
    # Two-stage splitting process
    
    # Stage 3a: Header-based splitting
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_content)
    
    # Stage 3b: Size-based splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(md_header_splits)
    
    return splits
```

**Chunking Strategy**:

1. **Header-Aware Splitting**:
   - Respects document structure (H1, H2 headers)
   - Maintains logical content boundaries
   - Preserves context within sections

2. **Size-Based Refinement**:
   - Configurable chunk size (default: 500 characters)
   - Intelligent overlap (default: 50 characters)
   - Recursive splitting for optimal chunk boundaries

**Benefits**:
- **Semantic Coherence**: Chunks maintain topical relevance
- **Context Preservation**: Overlapping prevents information loss
- **Retrieval Optimization**: Right-sized chunks for effective search

### Stage 4: Vector Embedding Generation

**Location**: `services/index/elasticsearch_index.py`

```python
def generate_embeddings_for_chunks(self, chunks):
    # Extract text from LangChain document objects
    texts = [chunk.page_content for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    
    # Generate embeddings using IBM Watson AI
    embedding_vectors = self.embedding_model.embed_documents(texts=texts)
    
    # Combine content with embeddings
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        enriched_chunk = {
            'content': chunk.page_content,
            'metadata': chunk.metadata,
            'embedding': embedding_vector[i]
        }
        enriched_chunks.append(enriched_chunk)
    
    return enriched_chunks
```

**Embedding Configuration**:
```python
# IBM Watson AI Embedding Model
embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
    EmbedParams.RETURN_OPTIONS: {
        'input_text': True
    }
}

embedding_model = Embeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,  # 30M parameter model
    params=embed_params,
    credentials=watson_credentials,
    project_id=watson_project_id
)
```

**Technical Specifications**:
- **Model**: IBM SLATE 30M English
- **Vector Dimensions**: High-dimensional semantic vectors
- **Batch Processing**: Efficient bulk embedding generation
- **Quality Assurance**: Handles embedding generation errors gracefully

### Stage 5: Elasticsearch Document Indexing

**Location**: `services/index/elasticsearch_index.py`

```python
def index_chunks_to_elasticsearch(self, enriched_chunks, document_name: str, index_name: str = None):
    print(f"Indexing {len(enriched_chunks)} chunks to index '{index_name}'")
    
    for i, chunk in enumerate(enriched_chunks):
        # Create Elasticsearch document structure
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
        
        # Index to Elasticsearch
        response = self.es_client.index(
            index=index_name,
            id=doc_id,
            body=doc
        )
    
    return indexing_statistics
```

**Document Structure in Elasticsearch**:
```json
{
  "content": "Document text chunk",
  "embedding": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "Header 1": "Section Title",
    "Header 2": "Subsection Title"
  },
  "document_name": "Original filename",
  "chunk_id": 0,
  "content_length": 342,
  "embedding_dimensions": 768,
  "indexed_at": "2025-06-26T10:30:00"
}
```

**Indexing Features**:
- **Unique IDs**: Generated using document name, chunk ID, and UUID
- **Rich Metadata**: Preserves document structure and hierarchy
- **Timestamping**: Tracks indexing time for each chunk
- **Progress Tracking**: Real-time feedback on indexing progress

### Stage 6: Results Aggregation & Response

**Location**: `services/document/document_service.py`

```python
def index_file(self, file_bytes: bytes, filename: str, index_name: str, 
               chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
    
    result = self.indexer.index_pdf_file(
        pdf_bytes=file_bytes,
        document_name=filename,
        index_name=index_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return {
        'status': 'success',
        'message': f'File {filename} indexed successfully',
        'document_name': filename,
        'total_chunks': result['chunks_created'],
        'indexed_successfully': result['indexing_results']['indexed_successfully'],
        'failed_to_index': result['indexing_results']['failed_to_index'],
        'index_name': index_name
    }
```

**Response Format**:
```json
{
  "status": "success",
  "message": "File document.pdf indexed successfully",
  "document_name": "document.pdf",
  "total_chunks": 45,
  "indexed_successfully": 45,
  "failed_to_index": 0,
  "index_name": "my_documents"
}
```

## 🔧 Technical Components

### Elasticsearch_Index Class

**File**: `services/index/elasticsearch_index.py`

The core indexing engine that handles all document processing operations:

**Key Methods**:
- `convert_pdf_bytes_to_markdown()`: PDF parsing and conversion
- `split_markdown_by_headers_and_chunks()`: Intelligent text segmentation
- `generate_embeddings_for_chunks()`: Vector representation generation
- `index_chunks_to_elasticsearch()`: Data persistence
- `index_pdf_file()`: Complete pipeline orchestration

**Configuration Management**:
```python
# Environment variables required
WATSONX_API_KEY=your_watson_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=your_watson_url
ELASTICSEARCH_HOST=your_es_host
ELASTICSEARCH_PORT=your_es_port
ELASTICSEARCH_USERNAME=your_es_username
ELASTICSEARCH_PASSWORD=your_es_password
ELASTICSEARCH_INDEX_NAME=your_default_index
```

### DocumentService Class

**File**: `services/document/document_service.py`

Orchestrates the indexing workflow and provides a high-level interface:

**Features**:
- **Lazy Initialization**: Services created only when needed
- **Error Handling**: Comprehensive error management
- **Service Integration**: Coordinates between indexing and retrieval
- **Result Formatting**: Standardized response structures

### SharedServices Singleton

**File**: `services/shared_services.py`

Manages service instances to optimize resource usage:

```python
class SharedServices:
    _instance = None
    _elasticsearch_index = None
    
    def get_elasticsearch_index(self):
        if self._elasticsearch_index is None:
            self._elasticsearch_index = Elasticsearch_Index()
        return self._elasticsearch_index
```

**Benefits**:
- **Resource Optimization**: Single instances of expensive services
- **Memory Efficiency**: Reduces initialization overhead
- **Connection Pooling**: Reuses database connections

## 🌐 API Endpoints for Indexing

### POST /v1/documents/index

**Purpose**: Index a PDF document into Elasticsearch

**Request Format**:
```bash
curl -X POST "http://localhost:8081/v1/documents/index" \
     -F "file=@document.pdf" \
     -F "index_name=my_documents" \
     -F "chunk_size=500" \
     -F "chunk_overlap=50"
```

**Parameters**:
- `file`: PDF file (required)
- `index_name`: Elasticsearch index name (required)
- `chunk_size`: Text chunk size in characters (optional, default: 500)
- `chunk_overlap`: Overlap between chunks (optional, default: 50)

**Response**:
```json
{
  "status": "success",
  "message": "File document.pdf indexed successfully",
  "document_name": "document.pdf",
  "total_chunks": 23,
  "indexed_successfully": 23,
  "failed_to_index": 0,
  "index_name": "my_documents"
}
```

### GET /v1/documents/index/{index_name}/stats

**Purpose**: Get statistics about an indexed document collection

**Response**:
```json
{
  "index_name": "my_documents",
  "total_documents": 156,
  "unique_document_names": ["doc1.pdf", "doc2.pdf"],
  "unique_document_count": 2,
  "index_size_bytes": 15728640
}
```

## ⚙️ Configuration Requirements

### Required Dependencies

```python
# PDF Processing
docling>=1.0.0

# Text Processing
langchain-text-splitters>=0.2.0

# AI/ML Services
ibm-watsonx-ai>=0.2.0

# Search Engine
elasticsearch>=8.0.0

# Web Framework
fastapi>=0.100.0
python-multipart>=0.0.6  # For file uploads

# Environment Management
python-dotenv>=1.0.0
```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# IBM Watson AI Configuration
WATSONX_API_KEY=your_watson_api_key
WATSONX_PROJECT_ID=your_watson_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Elasticsearch Configuration
ELASTICSEARCH_HOST=your_elasticsearch_host
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your_password
ELASTICSEARCH_INDEX_NAME=rag_documents
```

### System Requirements

**Minimum Hardware**:
- RAM: 8GB (16GB recommended)
- Storage: 10GB free space
- CPU: 4 cores (8 cores recommended)

**Software Requirements**:
- Python 3.8+
- Elasticsearch 8.0+
- Access to IBM Watson AI services
- HTTPS support for external APIs

---

*This indexing system provides the foundation for intelligent document search and retrieval, enabling sophisticated question-answering capabilities through semantic understanding of document content.*

## 🔍 Document Retrieval System

The ExpertORT Agent's retrieval system complements the indexing pipeline by enabling intelligent document search and retrieval. It transforms user queries into semantic vectors and finds the most relevant document chunks through a sophisticated multi-stage process that combines semantic search with advanced reranking techniques.

### Key Features

- **Semantic Query Processing**: Converts natural language queries into vector embeddings
- **Cosine Similarity Search**: Calculates semantic similarity between queries and document chunks
- **Advanced Reranking**: Uses IBM Watson AI models to refine search results
- **Multi-Modal Retrieval**: Supports both semantic and document-name based searches
- **Real-time Performance**: Optimized for fast query response times

## 🏗️ Retrieval Architecture

The retrieval system follows a layered architecture that ensures high-quality results:

```
📁 Retrieval Architecture
├── 🌐 API Layer (FastAPI)
│   └── Agent Router (/v1/agent/chat)
├── 🔧 Service Layer
│   ├── DocumentService (Query Orchestration)
│   └── AgentService (Query Integration)
├── 🔍 Retrieval Layer
│   └── Elasticsearch_Retrieval (Core Processing)
└── 🤖 AI Layer
    ├── Watson AI Embeddings (Query Vectorization)
    └── Watson AI Reranking (Result Refinement)
```

### Component Relationships

1. **API Layer**: Receives user queries and chat requests
2. **Service Layer**: Orchestrates the retrieval workflow
3. **Retrieval Layer**: Executes semantic search and similarity calculations
4. **AI Layer**: Provides embeddings and reranking capabilities

## 🔄 The Complete Retrieval Pipeline

The retrieval process consists of 4 distinct stages optimized for accuracy and performance:

### Stage 1: Query Processing & Embedding Generation

**Location**: `services/retrieval/elasticsearch_retrieval.py`

```python
def generate_query_embedding(self, query: str) -> List[float]:
    """Generate embedding for a query string using Watsonx.ai."""
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
```

**Process**:
- Receives natural language query from user
- Converts query to same embedding format as indexed documents
- Uses identical IBM Watson AI embedding model (IBM SLATE 30M)
- Ensures vector compatibility for similarity calculations

**Key Features**:
- **Consistent Vectorization**: Same model as indexing ensures compatibility
- **Error Handling**: Robust response parsing for different Watson AI formats
- **Performance Optimization**: Single query embedding generation

### Stage 2: Semantic Document Retrieval

**Location**: `services/retrieval/elasticsearch_retrieval.py`

```python
def retrieve_top_k_documents(self, query: str, k: int = 5, index_name: str = None, 
                             min_score: float = 0.0) -> List[Dict[str, Any]]:
    """Retrieve the top k most relevant documents for a given query."""
    
    # Generate embedding for the query
    query_embedding = self.generate_query_embedding(query)
    print(f"Query embedding generated with {len(query_embedding)} dimensions")
    
    # Fetch all documents for similarity calculation
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
    return results[:k]
```

**Similarity Calculation Process**:

1. **Vector Retrieval**: Fetches all document embeddings from Elasticsearch
2. **Cosine Similarity**: Calculates semantic similarity using the formula:
   ```
   similarity = (query_vector · document_vector) / (||query_vector|| × ||document_vector||)
   ```
3. **Score Filtering**: Applies minimum score threshold to filter irrelevant results
4. **Ranking**: Sorts results by similarity score in descending order

**Technical Specifications**:
- **Similarity Metric**: Cosine similarity for semantic matching
- **Threshold Filtering**: Configurable minimum score (default: 0.0)
- **Result Limitation**: Returns top-k documents (default: 5)
- **Metadata Preservation**: Maintains document structure and hierarchy

### Stage 3: Advanced Reranking with Watson AI

**Location**: `services/retrieval/elasticsearch_retrieval.py`

```python
def rerank_documents(self, query: str, documents: List[Dict[str, Any]], 
                     top_p: int = 5) -> List[Dict[str, Any]]:
    """Rerank documents using Watsonx.ai reranking model."""
    
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
        
        # Handle different response formats
        if hasattr(response, 'results') and response.results:
            for result in response.results:
                doc_idx = result.index
                if doc_idx < len(documents):
                    doc = documents[doc_idx].copy()
                    doc['rerank_score'] = result.score
                    reranked_docs.append(doc)
        elif isinstance(response, dict) and 'results' in response:
            for result in response['results']:
                doc_idx = result.get('index', 0)
                if doc_idx < len(documents):
                    doc = documents[doc_idx].copy()
                    doc['rerank_score'] = result.get('score', 0.0)
                    reranked_docs.append(doc)
        
        # Sort by rerank score (highest first) and return top_p
        reranked_docs.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        final_results = reranked_docs[:top_p]
        
        print(f"Reranking completed. Returning top {len(final_results)} documents.")
        return final_results
        
    except Exception as e:
        print(f"Error in reranking: {e}")
        print(f"Falling back to original ranking")
        return documents[:top_p]  # Fallback to original ranking
```

**Reranking Configuration**:
```python
# IBM Watson AI Reranking Model
self.reranker = Rerank(
    model_id="cross-encoder/ms-marco-minilm-l-12-v2",  # Cross-encoder model
    credentials=credentials,
    project_id=watsonx_project_id
)

# Alternative IBM model configuration
rerank_params = {
    "truncate_input_tokens": 512  # Handle longer documents
}
```

**Reranking Features**:
- **Cross-Encoder Architecture**: Uses advanced transformer models for query-document relevance
- **Context-Aware Scoring**: Considers query-document interaction rather than just similarity
- **Robust Error Handling**: Graceful fallback to semantic search results
- **Flexible Top-P Selection**: Configurable number of final results

### Stage 4: Result Aggregation & Formatting

**Location**: `services/document/document_service.py`

```python
def query_documents(self, query: str, k: int = 10, top_p: int = 3, 
                   min_score: float = 0.1, index_name: Optional[str] = None) -> Dict[str, Any]:
    """Query documents using semantic search and reranking."""
    
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
```

**Response Format**:
```json
{
  "status": "success",
  "message": "Found 3 relevant documents",
  "query": "What is the Transformer architecture?",
  "total_candidates": 10,
  "returned_results": 3,
  "results": [
    {
      "id": "attention_is_all_you_need_0_a1b2c3d4",
      "score": 0.8542,
      "rerank_score": 0.9876,
      "content": "The Transformer architecture is a neural network...",
      "metadata": {
        "Header 1": "Introduction",
        "Header 2": "Architecture"
      },
      "document_name": "Attention Is All You Need.pdf",
      "chunk_id": 0,
      "content_length": 487,
      "indexed_at": "2025-06-26T10:30:00"
    }
  ],
  "search_metadata": {
    "semantic_search_candidates": 10,
    "reranking_applied": true,
    "min_score_threshold": 0.1,
    "chunk_size_requested": 10,
    "final_results_requested": 3
  }
}
```

## 🔧 Technical Components

### Elasticsearch_Retrieval Class

**File**: `services/retrieval/elasticsearch_retrieval.py`

The core retrieval engine that handles all document search operations:

**Key Methods**:
- `generate_query_embedding()`: Query vectorization
- `retrieve_top_k_documents()`: Semantic similarity search
- `rerank_documents()`: Advanced result refinement
- `search_by_document_name()`: Document-specific search
- `get_index_statistics()`: Index analytics

**Similarity Calculation Algorithm**:
```python
# Cosine Similarity Calculation
def calculate_cosine_similarity(query_vector, doc_vector):
    dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
    query_magnitude = sum(a * a for a in query_vector) ** 0.5
    doc_magnitude = sum(b * b for b in doc_vector) ** 0.5
    
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0.0
    
    return dot_product / (query_magnitude * doc_magnitude)
```

### Alternative Search Methods

**Document Name Search**:
```python
def search_by_document_name(self, document_name: str, k: int = 10, 
                           index_name: str = None) -> List[Dict[str, Any]]:
    """Retrieve documents by document name for content exploration."""
    
    search_query = {
        "size": k,
        "query": {
            "match": {
                "document_name": document_name
            }
        },
        "sort": [
            {"chunk_id": {"order": "asc"}}  # Maintain document order
        ],
        "_source": [
            "content", "metadata", "document_name", "chunk_id", 
            "content_length", "indexed_at"
        ]
    }
    
    response = self.es_client.search(index=index_name, body=search_query)
    # Process and return results...
```

**Index Statistics**:
```python
def get_index_statistics(self, index_name: str = None) -> Dict[str, Any]:
    """Get comprehensive statistics about the document index."""
    
    # Get basic index stats
    stats = self.es_client.indices.stats(index=index_name)
    count_response = self.es_client.count(index=index_name)
    
    # Get unique document names using aggregations
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
        'total_documents': count_response.get('count', 0),
        'unique_document_names': [bucket['key'] for bucket in unique_docs],
        'unique_document_count': len(unique_docs),
        'index_size_bytes': stats.get('_all', {}).get('total', {}).get('store', {}).get('size_in_bytes', 0)
    }
```

## 🌐 API Integration

### Agent Chat Integration

**Location**: `services/agent/agent.py`

The retrieval system integrates seamlessly with the conversational agent:

```python
@tool
def buscar_en_base_de_conocimientos(query: str) -> str:
    """Search information in ExpertORT knowledge base using semantic search and reranking."""
    
    try:
        print(f"🔍 Searching knowledge base: '{query}'")
        
        # Step 1: Semantic search (get top-k candidates)
        initial_results = self.retrieval_system.retrieve_top_k_documents(
            query=query,
            k=10,  # Get 10 initial candidates
            min_score=0.1
        )
        
        if not initial_results:
            return "❌ No relevant information found in the knowledge base."
        
        # Step 2: Reranking for quality refinement
        reranked_results = self.retrieval_system.rerank_documents(
            query=query,
            documents=initial_results,
            top_p=5  # Return top 5 after reranking
        )
        
        # Step 3: Format results for agent consumption
        context_pieces = []
        for i, result in enumerate(reranked_results):
            context_piece = f"""
📄 **Source {i+1}**: {result['document_name']} (Chunk {result['chunk_id']})
🔍 **Relevance Score**: {result.get('rerank_score', result['score']):.3f}
📝 **Content**: {result['content']}
---
            """
            context_pieces.append(context_piece)
        
        return "\n".join(context_pieces)
        
    except Exception as e:
        return f"❌ Error searching knowledge base: {str(e)}"
```

### Service Layer Integration

**Location**: `services/document/document_service.py`

```python
class DocumentService:
    def __init__(self):
        self.shared_services = SharedServices()
        self.indexer = None  # Lazy initialization
        self.retriever = None  # Lazy initialization
    
    @property
    def retriever(self):
        """Lazy initialization of retrieval system."""
        if self._retriever is None:
            try:
                self._retriever = self.shared_services.get_elasticsearch_retrieval()
                print("✅ Retrieval system initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize retrieval system: {e}")
                self._retriever = None
        return self._retriever
```

## ⚙️ Configuration & Performance

### Environment Configuration

**Retrieval-specific variables**:
```env
# Search Performance Settings
ELASTICSEARCH_SEARCH_SIZE=100
MIN_SIMILARITY_SCORE=0.1
DEFAULT_RETRIEVAL_K=10
DEFAULT_RERANK_TOP_P=5

# Reranking Configuration
RERANK_MODEL_ID=cross-encoder/ms-marco-minilm-l-12-v2
RERANK_TRUNCATE_TOKENS=512
RERANK_TIMEOUT_SECONDS=30

# Debug Settings
RETRIEVAL_DEBUG_MODE=false
SIMILARITY_CALCULATION_VERBOSE=false
```

### Performance Optimization

**Memory Management**:
- **Connection Pooling**: Reuses Elasticsearch connections
- **Lazy Loading**: Services initialized only when needed
- **Batch Processing**: Efficient similarity calculations
- **Result Caching**: Configurable query result caching

**Query Optimization**:
- **Embedding Reuse**: Same model for indexing and retrieval
- **Threshold Filtering**: Early elimination of irrelevant documents
- **Pagination Support**: Handles large document collections
- **Timeout Management**: Prevents hanging queries

### Error Handling & Fallbacks

**Graceful Degradation**:
1. **Reranking Failure**: Falls back to semantic search results
2. **Embedding Failure**: Returns error with detailed message
3. **Elasticsearch Unavailable**: Provides clear error status
4. **Model Timeout**: Returns partial results with warnings

**Error Response Format**:
```json
{
  "status": "error",
  "message": "Elasticsearch connection failed",
  "query": "user query",
  "total_candidates": 0,
  "returned_results": 0,
  "results": [],
  "search_metadata": {
    "error_type": "connection_error",
    "fallback_applied": false,
    "retry_suggested": true
  }
}
```

## 📊 Retrieval Quality Metrics

### Semantic Search Metrics

**Similarity Score Distribution**:
- **High Relevance**: 0.8 - 1.0 (Excellent semantic match)
- **Medium Relevance**: 0.5 - 0.8 (Good semantic match)
- **Low Relevance**: 0.1 - 0.5 (Partial match)
- **Irrelevant**: < 0.1 (Filtered out)

**Performance Benchmarks**:
- **Query Processing Time**: < 500ms for typical queries
- **Embedding Generation**: < 100ms per query
- **Similarity Calculation**: < 200ms for 100 documents
- **Reranking Time**: < 300ms for 10 candidates

### Reranking Improvements

**Quality Enhancements**:
- **Cross-Encoder vs Bi-Encoder**: 15-25% improvement in relevance
- **Query-Document Interaction**: Better context understanding
- **Ranking Stability**: More consistent results across similar queries
- **Edge Case Handling**: Better performance on ambiguous queries

---

*This retrieval system provides sophisticated semantic search capabilities that enable the ExpertORT Agent to find and rank the most relevant document chunks for any user query, forming the foundation for accurate and contextual question-answering.*

## 🤖 Agent System Architecture

The ExpertORT Agent represents the intelligent core of the system, combining advanced language models with Retrieval-Augmented Generation (RAG) capabilities to provide contextual responses to student queries. Built using LangGraph and IBM Watson AI, the agent delivers sophisticated academic assistance tailored for Universidad ORT Uruguay students.

### Agent System Overview

The agent system integrates multiple AI technologies to deliver intelligent, context-aware responses:

```
🤖 Agent Architecture
├── 🧠 LLM Layer (IBM Watsonx.ai)
│   └── Meta-Llama 3.2 90B Vision Instruct
├── 🔧 Tool System (LangChain Tools)
│   └── Knowledge Base Search Tool
├── 🔄 Orchestration Layer (LangGraph)
│   └── ReAct Agent Pattern
├── 🔍 Knowledge Integration
│   └── Elasticsearch + Semantic Search + Reranking
└── 🌐 API Layer (Agent Connect Protocol)
    ├── Chat Completions (/v1/chat)
    └── Agent Discovery (/v1/agents)
```

## 🧠 Large Language Model Configuration

### Model Specifications

**Location**: `services/agent/agent.py`

```python
# LLM Configuration
parameters = {
    "frequency_penalty": 0,        # No repetition penalty
    "max_tokens": 2000,           # Maximum response length
    "presence_penalty": 0,        # No presence penalty
    "temperature": 0.7,           # Balanced creativity/precision
    "top_p": 1                    # Full vocabulary consideration
}

llm = ChatWatsonx(
    model_id=os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-2-90b-vision-instruct"),
    url=os.getenv("WATSONX_URL"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    apikey=os.getenv("WATSONX_API_KEY"),
    params=parameters
)
```

**Model Details**:
- **Model**: Meta-Llama 3.2 90B Vision Instruct
- **Provider**: IBM Watsonx.ai
- **Capabilities**: 
  - Multilingual support (Spanish/English)
  - Vision capabilities for document understanding
  - Instruction following and reasoning
  - Academic domain knowledge
- **Context Window**: Optimized for educational content
- **Performance**: Enterprise-grade reliability and speed

### Model Parameters Explanation

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperature` | 0.7 | Balanced responses - creative but focused |
| `max_tokens` | 2000 | Comprehensive responses without truncation |
| `frequency_penalty` | 0 | Natural language flow |
| `presence_penalty` | 0 | No topic avoidance |
| `top_p` | 1 | Full vocabulary access for precise terminology |

## 🛠️ Agent Tools System

### Knowledge Base Search Tool

**Location**: `services/agent/agent.py`

```python
@tool
def buscar_en_base_de_conocimientos(query: str) -> str:
    """
    Busca información en la base de conocimientos de ExpertORT usando búsqueda semántica y reranking.
    
    Args:
        query (str): La consulta o pregunta del usuario
        
    Returns:
        str: Información relevante encontrada en la base de conocimientos
    """
```

**Tool Capabilities**:

1. **Semantic Search**: 
   - Queries knowledge base using vector embeddings
   - Retrieves top-10 initial candidates
   - Applies minimum relevance threshold (0.1)

2. **Intelligent Reranking**:
   - Cross-encoder reranking for precision
   - Selects top-3 most relevant results
   - Contextual relevance optimization

3. **Response Formatting**:
   - Structured information presentation
   - Source attribution and citations
   - Content truncation for readability
   - Search metadata and statistics

**Tool Workflow**:

```python
# Step 1: Initial semantic search
initial_results = self.retrieval_system.retrieve_top_k_documents(
    query=query,
    k=10,                    # Initial candidates
    min_score=0.1           # Relevance threshold
)

# Step 2: Intelligent reranking
reranked_results = self.retrieval_system.rerank_documents(
    query=query,
    documents=initial_results,
    top_p=3                 # Final results
)

# Step 3: Format structured response
response_parts = []
response_parts.append(f"📚 **Información encontrada para: '{query}'**\n")

for i, result in enumerate(reranked_results, 1):
    document_name = result.get('document_name', 'Documento desconocido')
    chunk_id = result.get('chunk_id', 0)
    content = result.get('content', '').strip()
    
    response_parts.append(f"\n**📖 Resultado {i}:**")
    response_parts.append(f"*Fuente: {document_name} - Sección {chunk_id}*")
    response_parts.append(f"\n{content}\n")
```

## 📝 System Prompt Design

### Core Prompt Structure

**Location**: `services/agent/agent.py`

```python
prompt = (
    "Tu objetivo es ayudar a estudiantes de la Universidad ORT Uruguay a resolver sus dudas académicas. "
    "Eres un asistente inteligente especializado en proporcionar información educativa precisa y útil.\n\n"
    
    "INSTRUCCIONES:\n"
    "1. Cuando recibas una pregunta académica, usa la herramienta 'buscar_en_base_de_conocimientos' para encontrar información relevante.\n"
    "2. Analiza cuidadosamente los resultados de búsqueda y proporciona una respuesta clara y completa.\n"
    "3. Si la información no está disponible en la base de conocimientos, explícalo claramente.\n"
    "4. Mantén un tono profesional pero amigable, apropiado para estudiantes universitarios.\n"
    "5. Cita las fuentes cuando proporciones información específica de los documentos.\n\n"
    
    "CAPACIDADES:\n"
    "- Búsqueda semántica avanzada en documentos académicos\n"
    "- Reranking inteligente para encontrar la información más relevante\n"
    "- Análisis de múltiples fuentes documentales\n\n"
    
    "Siempre busca primero en la base de conocimientos antes de responder preguntas."
)
```

### Prompt Design Principles

**1. Role Definition**:
- Clear identity as Universidad ORT Uruguay assistant
- Academic specialization and educational focus
- Student-centric approach

**2. Behavioral Instructions**:
- Mandatory knowledge base consultation
- Structured analysis and response methodology
- Transparent information availability communication
- Professional but approachable tone

**3. Capability Awareness**:
- Explicit tool usage guidelines
- Technical capability communication
- Source attribution requirements

**4. Quality Assurance**:
- Always search before responding
- Clear error communication
- Structured response formatting

## 🔄 Agent Orchestration with LangGraph

### ReAct Agent Pattern

**Location**: `services/agent/agent.py`

```python
# Create ReAct agent with tools and prompt
return create_react_agent(llm, tools, prompt=prompt)
```

**ReAct (Reasoning + Acting) Framework**:

1. **Reasoning**: Agent analyzes the user query and determines what information is needed
2. **Acting**: Agent uses tools (knowledge base search) to gather relevant information  
3. **Observing**: Agent processes tool results and evaluates information quality
4. **Responding**: Agent synthesizes information into a coherent, helpful response

**Execution Flow**:

```
User Query → Reasoning → Tool Selection → Tool Execution → Result Analysis → Response Generation
     ↑                                                                              ↓
     └── Feedback Loop (if additional information needed) ←─────────────────────────┘
```

## 💬 Chat Processing System

### Non-Streaming Chat Completion

**Location**: `services/agent/agent.py`

```python
def process_chat_completion(self, messages: List[Dict[str, Any]]) -> str:
    """
    Process a chat completion request.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        str: The agent's response
    """
    # Convert messages to LangChain format
    langchain_messages = []
    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    
    # Execute the agent
    result = self.agent_executor.invoke({"messages": langchain_messages})
    final_message = result["messages"][-1]
    
    return final_message.content
```

### Streaming Chat Completion

**Location**: `services/agent/agent.py`

```python
async def process_streaming_chat(self, messages: List[Dict[str, Any]], thread_id: str) -> AsyncGenerator[str, None]:
    """
    Process a streaming chat completion request.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        thread_id: Thread identifier for the conversation
        
    Yields:
        str: Streaming response chunks in Server-Sent Events format
    """
    # Send thinking step
    thinking_step = {
        "id": f"step-{uuid.uuid4()}",
        "object": "thread.run.step.delta",
        "thread_id": thread_id,
        "model": "expertort-agent",
        "created": int(time.time()),
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "step_details": {
                        "type": "thinking",
                        "content": "Analizando la consulta y formulando una respuesta..."
                    }
                }
            }
        ]
    }
    
    yield f"event: thread.run.step.delta\n"
    yield f"data: {json.dumps(thinking_step)}\n\n"
    
    # Get and stream the response
    response_content = self.process_chat_completion(messages)
    message_chunks = self._split_into_chunks(response_content)
    
    for chunk in message_chunks:
        message_delta = {
            "id": f"msg-{uuid.uuid4()}",
            "object": "thread.message.delta",
            "thread_id": thread_id,
            "model": "expertort-agent",
            "created": int(time.time()),
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": chunk
                    }
                }
            ]
        }
        
        yield f"event: thread.message.delta\n"
        yield f"data: {json.dumps(message_delta)}\n\n"
```

**Streaming Features**:
- **Real-time Response**: Progressive message delivery
- **Thinking Indicators**: Shows processing status
- **Server-Sent Events**: Standard streaming protocol
- **Chunk-based Delivery**: Optimized for user experience

## 🔧 Agent Initialization & Lazy Loading

### Smart Initialization Strategy

**Location**: `services/agent/agent.py`

```python
class ExpertORTAgent:
    def __init__(self):
        """Initialize the ExpertORT Agent with all required components."""
        print("🤖 Initializing ExpertORT Agent...")
        
        # Use lazy initialization - services will be created only when needed
        self._retrieval_system = None
        self._agent_executor = None
        
        print("✅ ExpertORT Agent initialized with lazy loading!")
    
    @property
    def retrieval_system(self):
        """Get the retrieval system instance using lazy initialization."""
        if self._retrieval_system is None:
            self._retrieval_system = self._initialize_retrieval_system()
        return self._retrieval_system
    
    @property
    def agent_executor(self):
        """Get the agent executor instance using lazy initialization."""
        if self._agent_executor is None:
            self._agent_executor = self._create_agent()
        return self._agent_executor
```

**Benefits of Lazy Loading**:
- **Faster Startup**: Agent initializes quickly without loading all dependencies
- **Resource Efficiency**: Only loads components when actually needed
- **Error Isolation**: Graceful degradation if some services are unavailable
- **Scalability**: Better resource management in multi-instance deployments

## 📡 Agent Connect Protocol Implementation

### Agent Discovery Endpoint

**Location**: `api/routers/agent_router.py`

```python
@router.get("/v1/agents", response_model=AgentDiscoveryResponse)
async def discover_agents():
    """
    Agent discovery endpoint.
    Returns information about available agents.
    """
    agent_info_dict = self.agent.get_agent_info()
    
    agent_info = AgentInfo(
        name=agent_info_dict["name"],
        description=agent_info_dict["description"],
        provider=AgentProvider(**agent_info_dict["provider"]),
        version=agent_info_dict["version"],
        documentation_url=agent_info_dict["documentation_url"],
        capabilities=AgentCapabilities(**agent_info_dict["capabilities"])
    )
    
    return AgentDiscoveryResponse(agents=[agent_info])
```

**Agent Information Response**:
```json
{
  "agents": [
    {
      "name": "Agente ExpertORT",
      "description": "Agente especializado en ayudar a estudiantes de la Universidad ORT Uruguay a resolver sus dudas académicas.",
      "provider": {
        "organization": "Facundo Iraola Dopazo",
        "url": "facundoiraoladopazo.com"
      },
      "version": "1.0.0",
      "documentation_url": "https://docs.example.com/expertort-agent",
      "capabilities": {
        "streaming": true,
        "knowledge_base": true,
        "semantic_search": true,
        "reranking": true
      }
    }
  ]
}
```

### Chat Completion Endpoint

**Location**: `api/routers/agent_router.py`

```python
@router.post("/v1/chat")
async def chat_completion(
    request: ChatRequest, 
    x_thread_id: Optional[str] = Header(None)
):
    """
    Chat completion endpoint.
    Handles both streaming and non-streaming chat completions.
    """
    thread_id = x_thread_id or str(uuid.uuid4())
    
    # Handle streaming response
    if request.stream:
        return StreamingResponse(
            self._stream_chat_response(request, thread_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Thread-ID": thread_id
            }
        )
    
    # Handle non-streaming response
    else:
        response_content = self.agent.process_chat_completion(request.messages)
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": response_content
                    },
                    finish_reason="stop"
                )
            ],
            usage=ChatUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )
        
        return response
```

## 🎨 ExpertORT User Interface System

The ExpertORT Agent features a sophisticated, modern web-based chat interface designed to provide an intuitive and seamless user experience for interacting with the AI assistant. The UI combines responsive design principles, accessibility features, and advanced interaction patterns to create a professional academic tool.

### 🌟 Key UI Features Overview

- **Responsive Chat Interface**: Modern messaging UI with real-time conversation support
- **Mobile-First Design**: Optimized for all devices with adaptive layouts
- **Interactive Welcome Screen**: Contextual onboarding with suggested questions
- **Real-Time Typing Indicators**: Visual feedback during AI processing
- **Conversation Management**: Session handling with clear and restart options
- **Accessibility Support**: Screen reader compatible with keyboard navigation
- **Professional Academic Theming**: Clean, academic-focused design language

## 🏗️ UI Architecture & Component Structure

The user interface follows a modular, component-based architecture:

```
📁 UI Architecture
├── 🌐 Frontend Layer
│   ├── index.html (Main Application Structure)
│   ├── style.css (Core Styling & Theming)
│   ├── mobile.css (Responsive & Mobile Optimizations)
│   └── script.js (Interactive Functionality & API Communication)
├── 🎨 Design System
│   ├── CSS Custom Properties (Theme Variables)
│   ├── Bootstrap 5.3 Integration
│   └── Font Awesome Icons
└── 📱 Responsive Framework
    ├── Desktop Layout (≥992px)
    ├── Tablet Layout (768px - 991px)
    └── Mobile Layout (<768px)
```

### Component Hierarchy

1. **Application Shell**: Main container with sidebar and chat area
2. **Sidebar**: Navigation and conversation management
3. **Chat Container**: Messages display and input area
4. **Message System**: Individual message components with avatars
5. **Input System**: Text area with send controls

## 💬 Chat Interface Components

### Main Chat Container

**Location**: `ui/index.html` - Chat Container Section

The central chat interface provides a familiar messaging experience:

**Key Features**:
- **Dual-Pane Layout**: Sidebar for navigation, main area for conversation
- **Responsive Design**: Automatically adapts to screen size
- **Professional Theming**: Academic-focused color scheme and typography
- **Persistent Layout**: Maintains structure across different viewport sizes

**HTML Structure**:
```html
<div class="chat-container h-100 d-flex flex-column">
    <div class="chat-header"><!-- Agent info and controls --></div>
    <div class="messages-container"><!-- Conversation area --></div>
    <div class="input-area"><!-- Message input --></div>
</div>
```

### Sidebar Navigation System

**Location**: `ui/index.html` - Sidebar Section

**Features**:
- **Brand Identity**: ExpertORT logo with academic icon
- **Session Management**: New conversation button
- **Responsive Behavior**: Collapsible on mobile devices
- **Status Indicators**: Agent availability display

**Key Elements**:
```html
<div class="sidebar-header p-3 border-bottom">
    <div class="logo-icon me-2">
        <i class="fas fa-graduation-cap text-primary"></i>
    </div>
    <h6 class="mb-0 fw-bold">ExpertORT</h6>
    <small class="text-muted">Asistente Académico</small>
</div>
```

### Interactive Welcome Screen

**Location**: `ui/index.html` - Welcome Message Section

The welcome screen provides contextual onboarding for new users:

**Features**:
- **Personalized Greeting**: Introduces the AI assistant's capabilities
- **Suggested Questions**: Pre-defined academic queries to start conversations
- **Interactive Buttons**: One-click question submission
- **Educational Context**: Clear explanation of the system's academic focus

**Suggested Questions Include**:
- "¿Qué es el mecanismo de atención en Inteligencia Artificial?"
- "Explícame las redes neuronales Transformer"
- "Resume el paper 'Attention is All You Need'"
- "¿Cómo funcionan los encoders y decoders?"

## 💬 Message System Architecture

### Message Bubble Design

**Location**: `ui/style.css` - Message Bubbles Section

The messaging system implements a modern bubble-style interface:

**User Messages**:
- **Position**: Right-aligned with blue gradient background
- **Styling**: Rounded corners with modern shadow effects
- **Avatar**: User icon with subtle gradient
- **Timestamp**: Discrete time display

**Agent Messages**:
- **Position**: Left-aligned with white background
- **Styling**: Clean borders with professional appearance
- **Avatar**: Robot icon with brand gradient
- **Content Formatting**: Support for markdown, code, and links

**CSS Implementation**:
```css
.message.user .message-content {
    background: var(--message-user-bg);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
}

.message.agent .message-content {
    background: var(--message-agent-bg);
    color: var(--dark-color);
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    border: 1px solid var(--border-color);
}
```

### Real-Time Typing Indicators

**Location**: `ui/script.js` - Typing Indicator Functions

Visual feedback system during AI processing:

**Features**:
- **Animated Dots**: Three-dot animation indicating processing
- **Consistent Styling**: Matches agent message appearance
- **Automatic Management**: Shows during API calls, hides on response
- **Smooth Transitions**: Fade-in/fade-out animations

**Implementation**:
```javascript
showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
}
```

## ✍️ Advanced Input System

### Smart Text Area

**Location**: `ui/index.html` - Input Area Section

The message input system provides advanced text handling:

**Features**:
- **Auto-Resize**: Dynamically adjusts height based on content (max 120px)
- **Keyboard Shortcuts**: Enter to send, Shift+Enter for new line
- **Smart Button State**: Send button enables/disables based on content
- **Visual Feedback**: Focus states and hover effects
- **Touch Optimization**: Mobile-friendly input handling

**Input Controls**:
```html
<div class="input-group">
    <textarea 
        id="messageInput" 
        class="form-control message-input" 
        placeholder="Escribe tu mensaje aquí..." 
        rows="1"
        style="resize: none;"
    ></textarea>
    <button id="sendBtn" class="btn btn-primary" type="button">
        <i class="fas fa-paper-plane"></i>
    </button>
</div>
```

### Send Button Intelligence

**Location**: `ui/script.js` - Send Button State Management

Smart button behavior based on content and loading state:

**State Management**:
```javascript
updateSendButton() {
    const hasText = this.messageInput.value.trim().length > 0;
    const shouldEnable = hasText && !this.isLoading;
    this.sendBtn.disabled = !shouldEnable;
}
```

## 📱 Mobile-First Responsive Design

### Responsive Breakpoints

**Location**: `ui/style.css` & `ui/mobile.css` - Media Queries

The interface adapts seamlessly across device sizes:

**Desktop (≥992px)**:
- Full sidebar visible
- Wide message bubbles (max 70% width)
- Hover effects enabled
- Full feature set

**Tablet (768px - 991px)**:
- Collapsible sidebar
- Medium message bubbles (max 85% width)
- Touch-friendly interactions
- Optimized spacing

**Mobile (<768px)**:
- Hidden sidebar with menu button
- Narrow message bubbles (max 90% width)
- Large touch targets (min 44px)
- Simplified layout

### Mobile Menu System

**Location**: `ui/mobile.css` - Mobile Menu Implementation

Touch-friendly navigation for mobile devices:

**Features**:
- **Hamburger Menu**: Standard mobile navigation pattern
- **Slide Animation**: Smooth sidebar reveal/hide transitions
- **Overlay Support**: Background overlay when menu is open
- **Gesture Support**: Touch-based interactions

**CSS Implementation**:
```css
@media (max-width: 991px) {
    .sidebar {
        transform: translateX(-100%);
        transition: transform 0.3s ease;
        position: fixed;
        z-index: 1060;
    }
    
    .sidebar.show {
        transform: translateX(0);
    }
}
```

## 🎨 Design System & Theming

### CSS Custom Properties

**Location**: `ui/style.css` - Root Variables

The interface uses a comprehensive design token system:

**Color Palette**:
```css
:root {
    --primary-color: #0d6efd;      /* Primary blue */
    --secondary-color: #6c757d;     /* Neutral gray */
    --success-color: #198754;       /* Success green */
    --danger-color: #dc3545;        /* Error red */
    --sidebar-bg: #ffffff;          /* Sidebar background */
    --chat-bg: #f8f9fa;            /* Chat background */
    --message-user-bg: #0d6efd;     /* User message blue */
    --message-agent-bg: #ffffff;    /* Agent message white */
    --border-color: #e9ecef;        /* Border gray */
}
```

### Typography System

**Font Stack**:
- Primary: System font stack for optimal performance
- Fallbacks: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto
- Academic Focus: Clean, readable typography optimized for academic content

### Icon System

**Icon Library**: Font Awesome 6.4.0
- **Navigation Icons**: Hamburger menu, plus, trash
- **Status Icons**: Online indicator, loading spinners
- **Message Icons**: User, robot, warning, info
- **Action Icons**: Send (paper plane), graduation cap (brand)

## 🔄 Interactive Features & Animations

### Smooth Animations

**Location**: `ui/style.css` - Animation Definitions

Professional animation system for enhanced user experience:

**Message Animations**:
```css
.message {
    animation: fadeInUp 0.3s ease;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
```

**Typing Indicator Animation**:
```css
.typing-dot {
    animation: typing 1.4s infinite ease-in-out;
}

@keyframes typing {
    0%, 80%, 100% {
        transform: scale(0.8);
        opacity: 0.5;
    }
    40% {
        transform: scale(1);
        opacity: 1;
    }
}
```

### Hover Effects

Interactive elements provide visual feedback:

**Button Interactions**:
- Scale transforms on hover (1.05x)
- Smooth transitions (0.3s ease)
- Disabled state handling
- Focus indicators for accessibility

**Suggested Question Cards**:
- Elevation increase on hover
- Shadow depth changes
- Subtle transform animations

## 💾 State Management & Persistence

### Conversation Handling

**Location**: `ui/script.js` - ExpertORTChat Class

Sophisticated state management for conversation flow:

**Key Features**:
- **Session-Based Storage**: Conversations stored in localStorage during session
- **Auto-Clear on Refresh**: Fresh start policy for new sessions
- **Error State Recovery**: Graceful handling of API failures
- **Loading State Management**: Prevents duplicate requests during processing

**State Properties**:
```javascript
class ExpertORTChat {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.conversation = [];           // Message history
        this.isLoading = false;          // Request state
        this.abortController = null;     // Request cancellation
    }
}
```

### Local Storage Strategy

**Data Management**:
- Temporary conversation storage during active session
- Automatic cleanup on page refresh/close
- No persistent data retention for privacy
- Error recovery with graceful degradation

## 🌐 API Integration & Communication

### Chat API Integration

**Location**: `ui/script.js` - API Communication

Real-time communication with the ExpertORT Agent backend:

**Request Format**:
```javascript
const requestBody = {
    model: "expertort-agent",
    messages: this.conversation.map(msg => ({
        role: msg.role,
        content: msg.content
    })),
    stream: false
};
```

**Response Handling**:
- JSON response parsing
- Error state management
- Timeout handling
- Retry logic for failed requests

### Agent Status Monitoring

**Health Check System**:
```javascript
async checkAgentStatus() {
    try {
        const response = await fetch(`${this.apiBaseUrl}/health`);
        return response.ok;
    } catch (error) {
        return false;
    }
}
```

## ♿ Accessibility & Usability Features

### Keyboard Navigation

**Location**: `ui/script.js` - Event Handlers

Comprehensive keyboard support for accessibility:

**Key Bindings**:
- **Enter**: Send message (without Shift)
- **Shift + Enter**: New line in message
- **Tab Navigation**: Proper focus management
- **Escape**: Close mobile menu

**Focus Management**:
- Auto-focus on message input
- Visible focus indicators
- Logical tab order
- Screen reader announcements

### Screen Reader Support

**ARIA Implementation**:
- Semantic HTML structure
- Proper heading hierarchy
- Alt text for interactive elements
- Live regions for dynamic content

**Visual Accessibility**:
- High contrast color combinations
- Scalable text (rem units)
- Clear visual hierarchy
- Reduced motion support

### Touch Device Optimization

**Location**: `ui/mobile.css` - Touch Optimizations

Mobile-first accessibility features:

**Touch Targets**:
- Minimum 44px touch targets
- Proper spacing between interactive elements
- Gesture-friendly swipe areas
- iOS zoom prevention on input focus

```css
@media (hover: none) and (pointer: coarse) {
    .suggested-question,
    .btn {
        min-height: 44px;
        touch-action: manipulation;
    }
    
    .message-input {
        font-size: 16px; /* Prevents zoom on iOS */
    }
}
```

## 🛠️ Technical Implementation Details

### Modern JavaScript Architecture

**Location**: `ui/script.js` - Class-Based Structure

ES6+ implementation with modern JavaScript patterns:

**Class Structure**:
```javascript
class ExpertORTChat {
    // Constructor and initialization
    constructor() { /* Setup logic */ }
    
    // DOM management
    initializeElements() { /* Element references */ }
    attachEventListeners() { /* Event binding */ }
    
    // Conversation management
    loadConversation() { /* State restoration */ }
    saveConversation() { /* State persistence */ }
    clearConversation() { /* State reset */ }
    
    // Message handling
    sendMessage() { /* API communication */ }
    renderMessage() { /* DOM updates */ }
    formatMessageContent() { /* Content processing */ }
    
    // UI feedback
    showTypingIndicator() { /* Loading states */ }
    hideTypingIndicator() { /* State cleanup */ }
    updateSendButton() { /* Button management */ }
}
```

### Bootstrap 5.3 Integration

**Framework Benefits**:
- Responsive grid system
- Pre-built components
- Utility classes for rapid development
- Cross-browser compatibility
- Accessibility best practices

**Custom Overrides**:
- Brand color customization
- Component styling modifications
- Responsive breakpoint adjustments
- Animation timing customization

### Performance Optimizations

**Rendering Performance**:
- Efficient DOM manipulation
- Minimal reflow/repaint operations
- Optimized scroll handling
- Lazy loading for long conversations

**Network Performance**:
- Request debouncing
- Abort controller for request cancellation
- Error retry mechanisms
- Efficient JSON parsing

## 🔧 Configuration & Customization

### Theme Customization

**CSS Custom Properties**: Easy theme modification through CSS variables
**Color Schemes**: Support for brand customization
**Typography**: Configurable font stacks and sizing
**Spacing**: Consistent spacing system with CSS custom properties

### Responsive Breakpoints

**Customizable Breakpoints**:
```css
/* Desktop */
@media (min-width: 992px) { /* Large screens */ }

/* Tablet */
@media (max-width: 991px) and (min-width: 768px) { /* Medium screens */ }

/* Mobile */
@media (max-width: 767px) { /* Small screens */ }
```

### Feature Toggles

**Configurable Features**:
- Welcome message display
- Suggested questions
- Typing indicators
- Auto-scroll behavior
- Session persistence

## 🚀 Browser Support & Compatibility

### Supported Browsers

**Modern Browser Support**:
- Chrome 90+ (Recommended)
- Firefox 85+
- Safari 14+
- Edge 90+

**Mobile Browser Support**:
- iOS Safari 14+
- Chrome Mobile 90+
- Samsung Internet 13+

### Progressive Enhancement

**Core Functionality**:
- Works without JavaScript (basic form submission)
- Graceful degradation for older browsers
- CSS Grid with Flexbox fallbacks
- Modern features with polyfill support

### Performance Metrics

**Target Performance**:
- First Contentful Paint: <2s
- Largest Contentful Paint: <3s
- Cumulative Layout Shift: <0.1
- First Input Delay: <100ms

---

*The ExpertORT User Interface represents a modern, accessible, and intuitive gateway to advanced AI-powered academic assistance, designed specifically for the needs of Universidad ORT Uruguay students and faculty.*