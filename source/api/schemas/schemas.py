"""
API Schemas for the ExpertORT Agent.
Defines the data models used by the API endpoints following the Agent Connect protocol.
"""

from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class Message(BaseModel):
    """Message schema for chat interactions."""
    role: str
    content: str
    name: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat completion request schema."""
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False


class ChatChoice(BaseModel):
    """Individual choice in a chat completion response."""
    index: int
    message: Dict[str, str]
    finish_reason: str


class ChatUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response schema."""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage


class AgentProvider(BaseModel):
    """Agent provider information."""
    organization: str
    url: str


class AgentCapabilities(BaseModel):
    """Agent capabilities information."""
    streaming: bool
    knowledge_base: Optional[bool] = None
    semantic_search: Optional[bool] = None
    reranking: Optional[bool] = None


class AgentInfo(BaseModel):
    """Agent information schema."""
    name: str
    description: str
    provider: AgentProvider
    version: str
    documentation_url: str
    capabilities: AgentCapabilities


class AgentDiscoveryResponse(BaseModel):
    """Agent discovery response schema."""
    agents: List[AgentInfo]


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    message: str
    code: Optional[int] = None
