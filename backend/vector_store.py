"""
Vector Store Utility - Qdrant Cloud + OpenAI Embeddings
Handles document chunking, embedding generation, and vector storage for RAG.
"""

import os
import re
import hashlib
from typing import List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "physical_ai_docs")

# OpenAI embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

# Chunking configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks


@dataclass
class DocumentChunk:
    """Represents a chunk of document content."""
    id: str
    content: str
    metadata: dict
    embedding: Optional[List[float]] = None


class VectorStore:
    """Vector store manager using Qdrant Cloud and OpenAI embeddings."""

    def __init__(self):
        """Initialize Qdrant client and OpenAI client."""
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")

        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

        # Initialize OpenAI client
        self.openai = OpenAI(api_key=OPENAI_API_KEY)

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the collection if it doesn't exist."""
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if COLLECTION_NAME not in collection_names:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Created collection: {COLLECTION_NAME}")
        else:
            print(f"Collection already exists: {COLLECTION_NAME}")

    def _generate_chunk_id(self, content: str, source: str) -> str:
        """Generate a unique ID for a chunk based on content and source."""
        hash_input = f"{source}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _chunk_markdown(self, content: str, source: str) -> List[DocumentChunk]:
        """
        Split markdown content into overlapping chunks.
        Preserves markdown structure where possible.
        """
        chunks = []

        # Clean the content
        content = content.strip()

        # Split by headers first to preserve document structure
        # Match ## and ### headers
        header_pattern = r'(^#{2,3}\s+.+$)'
        sections = re.split(header_pattern, content, flags=re.MULTILINE)

        current_header = ""
        current_content = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Check if this is a header
            if re.match(r'^#{2,3}\s+', section):
                # Save previous section if it has content
                if current_content:
                    chunks.extend(
                        self._split_section(current_content, current_header, source)
                    )
                current_header = section
                current_content = ""
            else:
                current_content += "\n\n" + section if current_content else section

        # Don't forget the last section
        if current_content:
            chunks.extend(
                self._split_section(current_content, current_header, source)
            )

        return chunks

    def _split_section(
        self, content: str, header: str, source: str
    ) -> List[DocumentChunk]:
        """Split a section into chunks with overlap."""
        chunks = []
        content = content.strip()

        # If content is small enough, return as single chunk
        if len(content) <= CHUNK_SIZE:
            chunk_content = f"{header}\n\n{content}" if header else content
            chunks.append(
                DocumentChunk(
                    id=self._generate_chunk_id(chunk_content, source),
                    content=chunk_content,
                    metadata={
                        "source": source,
                        "header": header,
                        "chunk_index": 0,
                        "total_chunks": 1,
                    },
                )
            )
            return chunks

        # Split into overlapping chunks
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + CHUNK_SIZE

            # Try to break at a sentence or paragraph boundary
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind("\n\n", start, end)
                if para_break > start + CHUNK_SIZE // 2:
                    end = para_break
                else:
                    # Look for sentence break
                    sentence_break = content.rfind(". ", start, end)
                    if sentence_break > start + CHUNK_SIZE // 2:
                        end = sentence_break + 1

            chunk_text = content[start:end].strip()
            chunk_content = f"{header}\n\n{chunk_text}" if header else chunk_text

            chunks.append(
                DocumentChunk(
                    id=self._generate_chunk_id(chunk_content, source),
                    content=chunk_content,
                    metadata={
                        "source": source,
                        "header": header,
                        "chunk_index": chunk_index,
                    },
                )
            )

            # Move start position with overlap
            start = end - CHUNK_OVERLAP if end < len(content) else len(content)
            chunk_index += 1

        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text using OpenAI."""
        response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in a batch."""
        response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def ingest_markdown(
        self, content: str, source: str, batch_size: int = 100
    ) -> dict:
        """
        Ingest markdown content into the vector store.

        Args:
            content: Markdown content to ingest
            source: Source identifier (e.g., filename or URL)
            batch_size: Number of chunks to process in each batch

        Returns:
            Dict with ingestion statistics
        """
        # Chunk the content
        chunks = self._chunk_markdown(content, source)

        if not chunks:
            return {
                "status": "warning",
                "message": "No content to ingest",
                "chunks_created": 0,
            }

        # Generate embeddings in batches
        total_chunks = len(chunks)
        points = []

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]

            # Generate embeddings
            embeddings = self.generate_embeddings_batch(texts)

            # Create Qdrant points
            for chunk, embedding in zip(batch, embeddings):
                points.append(
                    PointStruct(
                        id=chunk.id,
                        vector=embedding,
                        payload={
                            "content": chunk.content,
                            **chunk.metadata,
                        },
                    )
                )

        # Upsert points to Qdrant
        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        return {
            "status": "success",
            "message": f"Successfully ingested {total_chunks} chunks from {source}",
            "chunks_created": total_chunks,
            "source": source,
        }

    async def search(
        self, query: str, limit: int = 5, score_threshold: float = 0.7
    ) -> List[dict]:
        """
        Search for relevant documents based on a query.

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of relevant document chunks with scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Search in Qdrant
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
        )

        return [
            {
                "content": hit.payload.get("content", ""),
                "source": hit.payload.get("source", ""),
                "header": hit.payload.get("header", ""),
                "score": hit.score,
            }
            for hit in results
        ]

    def get_collection_info(self) -> dict:
        """Get information about the vector collection."""
        collection = self.qdrant.get_collection(COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "vectors_count": collection.vectors_count,
            "points_count": collection.points_count,
            "status": collection.status.value,
        }

    def delete_collection(self) -> bool:
        """Delete the entire collection. Use with caution!"""
        self.qdrant.delete_collection(COLLECTION_NAME)
        return True

    def delete_by_source(self, source: str) -> dict:
        """Delete all chunks from a specific source."""
        self.qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source),
                        )
                    ]
                )
            ),
        )
        return {"status": "success", "message": f"Deleted all chunks from source: {source}"}


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
