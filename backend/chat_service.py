"""
RAG Chat Service - Core retrieval and generation logic
Handles semantic search and LLM-powered responses for the Physical AI documentation.
"""

import os
from typing import List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from vector_store import get_vector_store

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))


@dataclass
class ChatContext:
    """Represents context for a chat query."""
    content: str
    source: str
    score: float


@dataclass
class ChatResponse:
    """Response from the chat service."""
    answer: str
    sources: List[dict]
    context_used: str
    model: str


class ChatService:
    """RAG-powered chat service for Physical AI documentation."""

    def __init__(self):
        """Initialize the chat service."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")

        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.vector_store = get_vector_store()

    def _build_system_prompt(self, has_selected_context: bool = False) -> str:
        """Build the system prompt for the LLM."""
        base_prompt = """You are an expert AI assistant specializing in Physical AI and Humanoid Robotics.
You help users understand concepts from the "Physical AI and Humanoid Robotics" documentation.

Your role is to:
1. Answer questions accurately based on the provided context
2. Explain complex robotics and AI concepts clearly
3. Provide practical insights when relevant
4. Acknowledge when information is not available in the context

Guidelines:
- Base your answers primarily on the provided context
- Be concise but thorough
- Use technical terms appropriately but explain them when needed
- If the context doesn't contain enough information, say so clearly
- Format responses with markdown for readability when appropriate"""

        if has_selected_context:
            base_prompt += """

IMPORTANT: The user has selected specific text from the documentation.
Focus your answer primarily on explaining or expanding upon that selected text.
The selected text is provided in the "Selected Context" section below."""

        return base_prompt

    def _build_user_prompt(
        self,
        query: str,
        context_chunks: List[dict],
        selected_context: Optional[str] = None
    ) -> str:
        """Build the user prompt with context."""
        prompt_parts = []

        # Add selected context if provided (user-highlighted text)
        if selected_context:
            prompt_parts.append("## Selected Context (User-highlighted text)")
            prompt_parts.append(f"```\n{selected_context}\n```")
            prompt_parts.append("")

        # Add retrieved context chunks
        if context_chunks:
            prompt_parts.append("## Retrieved Documentation Context")
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk.get("source", "Unknown")
                content = chunk.get("content", "")
                score = chunk.get("score", 0)
                prompt_parts.append(f"### Source {i}: {source} (relevance: {score:.2f})")
                prompt_parts.append(content)
                prompt_parts.append("")

        # Add the user's question
        prompt_parts.append("## User Question")
        prompt_parts.append(query)

        return "\n".join(prompt_parts)

    async def chat(
        self,
        query: str,
        context_text: Optional[str] = None,
        max_chunks: Optional[int] = None,
        include_sources: bool = True
    ) -> ChatResponse:
        """
        Process a chat query using RAG.

        Args:
            query: The user's question
            context_text: Optional user-selected text from the frontend
            max_chunks: Maximum number of context chunks to retrieve
            include_sources: Whether to include source references

        Returns:
            ChatResponse with answer, sources, and metadata
        """
        max_chunks = max_chunks or MAX_CONTEXT_CHUNKS
        context_chunks = []
        context_type = "none"

        # If user provided selected context, use it as primary context
        if context_text and context_text.strip():
            context_type = "selected"
            # Still retrieve some chunks for additional context, but fewer
            try:
                retrieved = await self.vector_store.search(
                    query=query,
                    limit=max(1, max_chunks // 2),  # Fewer chunks when we have selected context
                    score_threshold=SIMILARITY_THRESHOLD
                )
                context_chunks = retrieved
            except Exception as e:
                print(f"Warning: Failed to retrieve additional context: {e}")
        else:
            # No selected context - rely entirely on vector search
            context_type = "retrieved"
            try:
                context_chunks = await self.vector_store.search(
                    query=query,
                    limit=max_chunks,
                    score_threshold=SIMILARITY_THRESHOLD
                )
            except Exception as e:
                print(f"Warning: Vector search failed: {e}")
                context_chunks = []

        # Build prompts
        system_prompt = self._build_system_prompt(has_selected_context=bool(context_text))
        user_prompt = self._build_user_prompt(
            query=query,
            context_chunks=context_chunks,
            selected_context=context_text
        )

        # Call OpenAI
        try:
            response = self.openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
            )

            answer = response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")

        # Prepare sources for response
        sources = []
        if include_sources and context_chunks:
            sources = [
                {
                    "source": chunk.get("source", "Unknown"),
                    "header": chunk.get("header", ""),
                    "score": round(chunk.get("score", 0), 3)
                }
                for chunk in context_chunks
            ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            context_used=context_type,
            model=CHAT_MODEL
        )

    async def chat_with_history(
        self,
        query: str,
        history: List[dict],
        context_text: Optional[str] = None,
        max_chunks: Optional[int] = None
    ) -> ChatResponse:
        """
        Process a chat query with conversation history.

        Args:
            query: The user's current question
            history: List of previous messages [{"role": "user"|"assistant", "content": "..."}]
            context_text: Optional user-selected text
            max_chunks: Maximum context chunks

        Returns:
            ChatResponse with answer and metadata
        """
        max_chunks = max_chunks or MAX_CONTEXT_CHUNKS
        context_chunks = []

        # Retrieve context based on the current query
        if context_text and context_text.strip():
            try:
                context_chunks = await self.vector_store.search(
                    query=query,
                    limit=max(1, max_chunks // 2),
                    score_threshold=SIMILARITY_THRESHOLD
                )
            except Exception:
                pass
        else:
            try:
                context_chunks = await self.vector_store.search(
                    query=query,
                    limit=max_chunks,
                    score_threshold=SIMILARITY_THRESHOLD
                )
            except Exception:
                pass

        # Build system prompt
        system_prompt = self._build_system_prompt(has_selected_context=bool(context_text))

        # Build messages array with history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (limit to last 10 exchanges)
        for msg in history[-20:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        # Add current query with context
        user_prompt = self._build_user_prompt(
            query=query,
            context_chunks=context_chunks,
            selected_context=context_text
        )
        messages.append({"role": "user", "content": user_prompt})

        # Call OpenAI
        try:
            response = self.openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )

            answer = response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")

        # Prepare sources
        sources = [
            {
                "source": chunk.get("source", "Unknown"),
                "header": chunk.get("header", ""),
                "score": round(chunk.get("score", 0), 3)
            }
            for chunk in context_chunks
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            context_used="selected" if context_text else "retrieved",
            model=CHAT_MODEL
        )


# Singleton instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create the chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
