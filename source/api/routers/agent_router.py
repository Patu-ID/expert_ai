"""
API Routers for the ExpertORT Agent.
Defines the endpoints that expose the agent's functionality following the Agent Connect protocol.
"""

import time
import uuid
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas.schemas import (
    ChatRequest, 
    ChatCompletionResponse, 
    ChatChoice, 
    ChatUsage, 
    AgentDiscoveryResponse,
    AgentInfo,
    AgentProvider,
    AgentCapabilities,
    ErrorResponse
)
from services.agent.agent import ExpertORTAgent


class AgentRouter:
    """Router class for ExpertORT Agent API endpoints."""
    
    def __init__(self, agent: ExpertORTAgent):
        """
        Initialize the router with an agent instance.
        
        Args:
            agent: ExpertORTAgent instance
        """
        self.agent = agent
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up the API routes."""
        
        @self.router.get("/v1/agents", response_model=AgentDiscoveryResponse)
        async def discover_agents():
            """
            Agent discovery endpoint.
            Returns information about available agents.
            """
            try:
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
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error retrieving agent information: {str(e)}")
        
        @self.router.post("/v1/chat")
        async def chat_completion(
            request: ChatRequest, 
            x_thread_id: Optional[str] = Header(None)
        ):
            """
            Chat completion endpoint.
            Handles both streaming and non-streaming chat completions.
            """
            thread_id = x_thread_id or str(uuid.uuid4())
            
            try:
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
                    
                    # Format the response according to the protocol
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
                            prompt_tokens=0,  # Could be calculated if needed
                            completion_tokens=0,  # Could be calculated if needed
                            total_tokens=0  # Could be calculated if needed
                        )
                    )
                    
                    return response
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")
    
    async def _stream_chat_response(self, request: ChatRequest, thread_id: str):
        """
        Stream chat response generator.
        
        Args:
            request: Chat completion request
            thread_id: Thread identifier
            
        Yields:
            str: Server-sent events data
        """
        try:
            async for chunk in self.agent.process_streaming_chat(request.messages, thread_id):
                yield chunk
            
            # Send completion signal
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            # Send error in streaming format
            error_data = {
                "error": "stream_error",
                "message": str(e)
            }
            yield f"data: {error_data}\n\n"
    
    def get_router(self) -> APIRouter:
        """
        Get the configured router.
        
        Returns:
            APIRouter: Configured FastAPI router
        """
        return self.router


def create_agent_router(agent: ExpertORTAgent) -> APIRouter:
    """
    Factory function to create an agent router.
    
    Args:
        agent: ExpertORTAgent instance
        
    Returns:
        APIRouter: Configured router
    """
    agent_router = AgentRouter(agent)
    return agent_router.get_router()
