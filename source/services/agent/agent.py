"""
ExpertORT Agent Service Layer.
Contains the business logic for the Watsonx.ai agent with RAG capabilities.
"""

import os
import json
import time
import uuid
from typing import Dict, List, Any, AsyncGenerator
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_ibm import ChatWatsonx
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

from services.shared_services import shared_services

# Load environment variables
load_dotenv()


class ExpertORTAgent:
    """
    ExpertORT Agent class that handles all business logic for the AI assistant.
    Integrates with Elasticsearch for knowledge base retrieval and Watsonx.ai for language processing.
    """
    
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
    
    def _initialize_retrieval_system(self):
        """Initialize the Elasticsearch retrieval system using shared services."""
        try:
            print("🔍 Getting shared Elasticsearch retrieval system...")
            retrieval_system = shared_services.get_elasticsearch_retrieval()
            print("✅ Retrieval system initialized successfully!")
            return retrieval_system
        except Exception as e:
            print(f"❌ Error initializing retrieval system: {e}")
            return None
            print("⚠️ Agent will continue without knowledge base access")
            return None
    
    def _create_knowledge_search_tool(self):
        """Create the knowledge base search tool."""
        @tool
        def buscar_en_base_de_conocimientos(query: str) -> str:
            """
            Busca información en la base de conocimientos de ExpertORT usando búsqueda semántica y reranking.
            
            Args:
                query (str): La consulta o pregunta del usuario
                
            Returns:
                str: Información relevante encontrada en la base de conocimientos
            """
            if not self.retrieval_system:
                return "❌ Error: Sistema de búsqueda no disponible. Verifica la configuración de Elasticsearch y Watsonx.ai."
            
            try:
                print(f"🔍 Buscando en base de conocimientos: '{query}'")
                
                # Step 1: Realizar búsqueda semántica (obtener top-k documentos)
                initial_results = self.retrieval_system.retrieve_top_k_documents(
                    query=query,
                    k=10,  # Obtener 10 candidatos iniciales
                    min_score=0.1  # Filtro mínimo de relevancia
                )
                
                if not initial_results:
                    return f"❌ No se encontró información relevante para: '{query}'. Intenta reformular tu pregunta."
                
                print(f"📄 Encontrados {len(initial_results)} documentos candidatos")
                
                # Step 2: Aplicar reranking para obtener los más relevantes
                reranked_results = self.retrieval_system.rerank_documents(
                    query=query,
                    documents=initial_results,
                    top_p=3  # Obtener los 3 más relevantes después del reranking
                )
                
                print(f"🎯 Reranking completado. Top {len(reranked_results)} resultados seleccionados")
                
                # Step 3: Formatear la respuesta con los resultados más relevantes
                if not reranked_results:
                    return f"❌ No se encontraron resultados relevantes después del reranking para: '{query}'. Intenta reformular tu pregunta."
                
                response_parts = []
                response_parts.append(f"📚 **Información encontrada para: '{query}'**\n")
                
                for i, result in enumerate(reranked_results, 1):
                    document_name = result.get('document_name', 'Documento desconocido')
                    chunk_id = result.get('chunk_id', 0)
                    content = result.get('content', '').strip()
                    
                    # Only show content if it's meaningful (not just headers)
                    if len(content) < 50:  # Skip very short content like headers
                        continue
                        
                    response_parts.append(f"\n**📖 Resultado {i}:**")
                    response_parts.append(f"*Fuente: {document_name} - Sección {chunk_id}*")
                    
                    # Truncate very long content to keep response manageable
                    if len(content) > 500:
                        content = content[:500] + "..."
                    
                    response_parts.append(f"\n{content}\n")
                    response_parts.append("---")
                
                # Add summary information
                response_parts.append(f"\n💡 **Resumen de búsqueda:**")
                response_parts.append(f"- Se analizaron {len(initial_results)} documentos candidatos")
                response_parts.append(f"- Se seleccionaron {len(reranked_results)} resultados más relevantes")
                response_parts.append(f"- Búsqueda realizada con IA semántica y reranking inteligente")
                
                return "\n".join(response_parts)
                
            except Exception as e:
                error_message = f"❌ Error durante la búsqueda: {str(e)}"
                print(error_message)
                return error_message
        
        return buscar_en_base_de_conocimientos
    
    def _create_agent(self):
        """Create the LangGraph agent with tools and configuration."""
        # Initialize the LLM
        parameters = {
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0.7,
            "top_p": 1
        }
        
        llm = ChatWatsonx(
            model_id=os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-2-90b-vision-instruct"),
            url=os.getenv("WATSONX_URL"),
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            apikey=os.getenv("WATSONX_API_KEY"),
            params=parameters
        )
        
        # Define the system prompt
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
        
        # Create tools
        tools = [self._create_knowledge_search_tool()]
        
        # Create and return the agent
        return create_react_agent(llm, tools, prompt=prompt)
    
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
    
    async def process_streaming_chat(self, messages: List[Dict[str, Any]], thread_id: str) -> AsyncGenerator[str, None]:
        """
        Process a streaming chat completion request.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            thread_id: Thread identifier for the conversation
            
        Yields:
            str: Streaming response chunks in Server-Sent Events format
        """
        # First, send a thinking step
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
        
        # Get the agent's response
        response_content = self.process_chat_completion(messages)
        
        # Stream the response in chunks
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
    
    def _split_into_chunks(self, text: str, chunk_size: int = 10) -> List[str]:
        """
        Split text into chunks for streaming.
        
        Args:
            text: Text to split
            chunk_size: Number of words per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent for discovery.
        
        Returns:
            Dict[str, Any]: Agent information dictionary
        """
        return {
            "name": "Agente ExpertORT",
            "description": "Agente especializado en ayudar a estudiantes de la Universidad ORT Uruguay a resolver sus dudas académicas.",
            "provider": {
                "organization": "Facundo Iraola Dopazo",
                "url": "facundoiraoladopazo.com",
            },
            "version": "1.0.0",
            "documentation_url": "https://docs.example.com/expertort-agent",
            "capabilities": {
                "streaming": True,
                "knowledge_base": self.retrieval_system is not None,
                "semantic_search": True,
                "reranking": True
            }
        }
