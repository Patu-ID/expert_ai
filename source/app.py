"""
ExpertORT Agent Application.
Main FastAPI application that initializes and serves the ExpertORT Agent API.
"""

import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from services.agent.agent import ExpertORTAgent
from services.document.document_service import DocumentService
from api.routers.agent_router import create_agent_router
from api.routers.document_router import create_document_router


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    # Create FastAPI app
    app = FastAPI(
        title="ExpertORT Agent API",
        description="Agente inteligente para estudiantes de la Universidad ORT Uruguay",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize the agent
    print("🚀 Initializing ExpertORT Agent...")
    agent = ExpertORTAgent()
    
    # Initialize the document service
    print("📚 Initializing Document Service...")
    document_service = DocumentService()
    
    # Create and include the agent router
    agent_router = create_agent_router(agent)
    app.include_router(agent_router, tags=["agent"])
    
    # Create and include the document router
    document_router = create_document_router(document_service)
    app.include_router(document_router, tags=["documents"])
    
    # Mount static files for the UI
    ui_path = os.path.join(current_dir, "ui")
    if os.path.exists(ui_path):
        app.mount("/static", StaticFiles(directory=ui_path), name="static")
        
        # Serve the main UI at the root
        @app.get("/chat")
        async def serve_chat_ui():
            """Serve the chat UI."""
            return FileResponse(os.path.join(ui_path, "index.html"))
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "ExpertORT Agent",
            "version": "1.0.0",
            "knowledge_base_available": agent._retrieval_system is not None,
            "document_indexing_available": document_service._indexer is not None,
            "document_retrieval_available": document_service._retriever is not None
        }
    
    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with basic information."""
        return {
            "message": "ExpertORT Agent API",
            "description": "Agente inteligente para estudiantes de la Universidad ORT Uruguay",
            "version": "1.0.0",
            "ui": {
                "chat_interface": "/chat",
                "documentation": "/docs"
            },
            "endpoints": {
                "agents": "/v1/agents",
                "chat": "/v1/chat",
                "index_file": "/v1/documents/index",
                "query_documents": "/v1/documents/query",
                "index_stats": "/v1/documents/index/{index_name}/stats",
                "search_document": "/v1/documents/search/{document_name}",
                "health": "/health",
                "docs": "/docs"
            }
        }
    
    print("✅ ExpertORT Agent API initialized successfully!")
    return app


# Create the app instance - only when running as main module
app = None


def get_app():
    """
    Get or create the FastAPI app instance.
    Implements lazy initialization to prevent multiple app creation.
    """
    global app
    if app is None:
        app = create_app()
    return app


def main():
    """
    Main function to run the application.
    """
    app_instance = get_app()
    
    print("🌟 Starting ExpertORT Agent Server...")
    print("🎨 Chat UI available at: http://localhost:8082/chat")
    print("📚 Access documentation at: http://localhost:8082/docs")
    print("🔍 Health check at: http://localhost:8082/health")
    print("🤖 Agent discovery at: http://localhost:8082/v1/agents")
    print("💬 Chat endpoint at: http://localhost:8082/v1/chat")
    print("📄 Index files at: http://localhost:8082/v1/documents/index")
    print("🔍 Query documents at: http://localhost:8082/v1/documents/query")
    
    uvicorn.run(
        app_instance,  # Use the app instance directly instead of string reference
        host="0.0.0.0",
        port=8082,  # Use different port to avoid conflicts
        reload=False,  # Disable auto-reload to prevent multiple initialization
        log_level="info"
    )


if __name__ == "__main__":
    main()
else:
    # For uvicorn imports - create app instance only when imported as module
    app = get_app()
