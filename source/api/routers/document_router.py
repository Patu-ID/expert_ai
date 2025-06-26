"""
Document API Router for the ExpertORT Agent.
Defines endpoints for document indexing and querying.
"""

import io
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

from api.schemas.schemas import (
    FileIndexRequest,
    FileIndexResponse, 
    QueryRequest,
    QueryResponse,
    DocumentResult,
    ErrorResponse
)
from services.document.document_service import DocumentService


class DocumentRouter:
    """Router class for document-related API endpoints."""
    
    def __init__(self, document_service: DocumentService):
        """
        Initialize the router with a document service instance.
        
        Args:
            document_service: DocumentService instance
        """
        self.document_service = document_service
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up the API routes."""
        
        @self.router.post("/v1/documents/index", response_model=FileIndexResponse)
        async def index_file(
            file: UploadFile = File(...),
            index_name: str = Form(...),
            chunk_size: Optional[int] = Form(500),
            chunk_overlap: Optional[int] = Form(50)
        ):
            """
            Index a PDF file to Elasticsearch.
            
            Args:
                file: PDF file to index
                index_name: Name of the Elasticsearch index
                chunk_size: Size of text chunks (characters)
                chunk_overlap: Overlap between chunks (characters)
                
            Returns:
                FileIndexResponse: Indexing results
            """
            try:
                # Validate file type
                if not file.filename.lower().endswith('.pdf'):
                    raise HTTPException(
                        status_code=400, 
                        detail="Only PDF files are supported"
                    )
                
                # Read file content
                file_content = await file.read()
                
                if not file_content:
                    raise HTTPException(
                        status_code=400,
                        detail="File is empty"
                    )
                
                print(f"📄 Received file: {file.filename} ({len(file_content)} bytes)")
                
                # Process the file
                result = self.document_service.index_file(
                    file_bytes=file_content,
                    filename=file.filename,
                    index_name=index_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                return FileIndexResponse(**result)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing file: {str(e)}"
                )
        
        @self.router.post("/v1/documents/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            """
            Query documents using semantic search and reranking.
            
            Args:
                request: Query request with search parameters
                
            Returns:
                QueryResponse: Search results with metadata
            """
            try:
                if not request.query.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="Query cannot be empty"
                    )
                
                print(f"🔍 Processing query: '{request.query}'")
                
                # Process the query
                result = self.document_service.query_documents(
                    query=request.query,
                    k=request.k,
                    top_p=request.top_p,
                    min_score=request.min_score,
                    index_name=request.index_name
                )
                
                # Convert results to the proper schema format
                if result['status'] == 'success':
                    formatted_results = []
                    for doc in result['results']:
                        doc_result = DocumentResult(**doc)
                        formatted_results.append(doc_result)
                    
                    result['results'] = formatted_results
                
                return QueryResponse(**result)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing query: {str(e)}"
                )
        
        @self.router.get("/v1/documents/index/{index_name}/stats")
        async def get_index_statistics(index_name: str):
            """
            Get statistics about an Elasticsearch index.
            
            Args:
                index_name: Name of the index
                
            Returns:
                Dict: Index statistics
            """
            try:
                stats = self.document_service.get_index_statistics(index_name)
                
                if 'error' in stats:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error getting index statistics: {stats['error']}"
                    )
                
                return stats
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error getting index statistics: {str(e)}"
                )
        
        @self.router.get("/v1/documents/search/{document_name}")
        async def search_by_document_name(
            document_name: str,
            k: Optional[int] = 10,
            index_name: Optional[str] = None
        ):
            """
            Search for chunks of a specific document by name.
            
            Args:
                document_name: Name of the document to search for
                k: Number of chunks to return
                index_name: Name of the index (optional)
                
            Returns:
                List: Document chunks
            """
            try:
                results = self.document_service.search_by_document_name(
                    document_name=document_name,
                    k=k,
                    index_name=index_name
                )
                
                return {
                    "document_name": document_name,
                    "total_chunks": len(results),
                    "chunks": results
                }
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error searching for document: {str(e)}"
                )
    
    def get_router(self) -> APIRouter:
        """
        Get the configured router.
        
        Returns:
            APIRouter: Configured FastAPI router
        """
        return self.router


def create_document_router(document_service: DocumentService) -> APIRouter:
    """
    Factory function to create a document router.
    
    Args:
        document_service: DocumentService instance
        
    Returns:
        APIRouter: Configured router
    """
    document_router = DocumentRouter(document_service)
    return document_router.get_router()
