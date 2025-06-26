"""
API Schemas for the ExpertORT Agent.
Defines the data models used by the API endpoints following the Agent Connect protocol.
"""

from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from fastapi import UploadFile


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


class FileIndexRequest(BaseModel):
    """File indexing request schema."""
    index_name: str
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50


class FileIndexResponse(BaseModel):
    """File indexing response schema."""
    status: str
    message: str
    document_name: str
    total_chunks: Optional[int] = None
    indexed_successfully: Optional[int] = None
    failed_to_index: Optional[int] = None
    index_name: Optional[str] = None


class QueryRequest(BaseModel):
    """Query request schema."""
    query: str
    k: Optional[int] = 10
    top_p: Optional[int] = 3
    min_score: Optional[float] = 0.1
    index_name: Optional[str] = None


class DocumentResult(BaseModel):
    """Individual document result schema."""
    id: str
    score: float
    rerank_score: Optional[float] = None
    content: str
    metadata: Dict[str, Any]
    document_name: str
    chunk_id: int
    content_length: int
    indexed_at: str


class QueryResponse(BaseModel):
    """Query response schema."""
    status: str
    query: str
    total_candidates: int
    returned_results: int
    results: List[DocumentResult]
    search_metadata: Dict[str, Any]
