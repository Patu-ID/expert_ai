"""
ExpertORT Agent Application.
Main FastAPI application that initializes and serves the ExpertORT Agent API.
"""

import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from services.agent.agent import ExpertORTAgent
from api.routers.agent_router import create_agent_router


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
    
    # Create and include the agent router
    agent_router = create_agent_router(agent)
    app.include_router(agent_router, tags=["agent"])
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "ExpertORT Agent",
            "version": "1.0.0",
            "knowledge_base_available": agent.retrieval_system is not None
        }
    
    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with basic information."""
        return {
            "message": "ExpertORT Agent API",
            "description": "Agente inteligente para estudiantes de la Universidad ORT Uruguay",
            "version": "1.0.0",
            "endpoints": {
                "agents": "/v1/agents",
                "chat": "/v1/chat",
                "health": "/health",
                "docs": "/docs"
            }
        }
    
    print("✅ ExpertORT Agent API initialized successfully!")
    return app


# Create the app instance
app = create_app()


def main():
    """
    Main function to run the application.
    """
    print("🌟 Starting ExpertORT Agent Server...")
    print("📚 Access documentation at: http://localhost:8081/docs")
    print("🔍 Health check at: http://localhost:8081/health")
    print("🤖 Agent discovery at: http://localhost:8081/v1/agents")
    print("💬 Chat endpoint at: http://localhost:8081/v1/chat")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8081,  # Use different port to avoid conflicts
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )


if __name__ == "__main__":
    main()
