"""
RAG Chatbot Backend - Gemini AI + Qdrant Vector Store
Physical AI and Humanoid Robotics Documentation Assistant
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Model configuration
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIMENSION = 768

# Collection name
COLLECTION_NAME = "physical_ai_docs"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# System instruction for AI personality
SYSTEM_INSTRUCTION = """You are a helpful assistant for the Physical AI & Humanoid Robotics book. You were created and developed by **Ubaid Ur Rehman**. If anyone asks who built you or created you, always proudly mention his name.

You specialize in:
- Physical AI (AI systems that interact with the physical world)
- Humanoid robotics (bipedal robots, locomotion, manipulation)
- Robot perception (computer vision, sensors, SLAM)
- Robot learning (reinforcement learning, imitation learning)
- Control systems for robots

Be helpful, accurate, and technically precise in your responses."""

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Qdrant client
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ==================== Gemini Model Discovery ====================

# Global cache for the working model
_gemini_model = None
_gemini_model_name = None


def discover_gemini_model():
    """
    Discover a working Gemini model that supports generateContent.
    Uses genai.list_models() to find available models.
    """
    global _gemini_model, _gemini_model_name

    if _gemini_model is not None:
        return _gemini_model, _gemini_model_name

    print("\n[GEMINI] Discovering available models...")

    try:
        # List all available models
        available_models = list(genai.list_models())
        print(f"[GEMINI] Found {len(available_models)} models")

        # Filter models that support generateContent
        content_models = []
        for model in available_models:
            supported_methods = getattr(model, 'supported_generation_methods', [])
            if 'generateContent' in supported_methods:
                content_models.append(model)
                print(f"[GEMINI]   - {model.name} (supports generateContent)")

        if not content_models:
            raise RuntimeError("No models found that support generateContent")

        # Prefer flash models, then pro models
        preferred_order = ['flash', 'pro', 'ultra']
        selected_model = None

        for preference in preferred_order:
            for model in content_models:
                if preference in model.name.lower():
                    selected_model = model
                    break
            if selected_model:
                break

        # Fallback to first available
        if not selected_model:
            selected_model = content_models[0]

        model_name = selected_model.name
        print(f"\n[GEMINI] Selected model: {model_name}")

        # Create the GenerativeModel instance with system instruction
        _gemini_model = genai.GenerativeModel(
            model_name,
            system_instruction=SYSTEM_INSTRUCTION
        )
        _gemini_model_name = model_name

        # Test the model
        print(f"[GEMINI] Testing model...")
        test_response = _gemini_model.generate_content("Say 'OK' if you're working.")
        if test_response and test_response.text:
            print(f"[GEMINI] Model test successful!")

        return _gemini_model, _gemini_model_name

    except Exception as e:
        print(f"[GEMINI] Error discovering models: {e}")

        # Fallback: try common model names directly
        fallback_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
            "models/gemini-1.5-flash",
            "models/gemini-pro",
        ]

        print(f"[GEMINI] Trying fallback models...")
        for model_name in fallback_models:
            try:
                print(f"[GEMINI]   Trying {model_name}...")
                model = genai.GenerativeModel(
                    model_name,
                    system_instruction=SYSTEM_INSTRUCTION
                )
                test = model.generate_content("Hi")
                if test and test.text:
                    print(f"[GEMINI] Fallback successful: {model_name}")
                    _gemini_model = model
                    _gemini_model_name = model_name
                    return _gemini_model, _gemini_model_name
            except Exception as fallback_err:
                print(f"[GEMINI]   Failed: {str(fallback_err)[:40]}")
                continue

        raise RuntimeError(
            "Could not find a working Gemini model. "
            "Please check your GEMINI_API_KEY and ensure it has access to Gemini models."
        )


def get_gemini_model():
    """Get the cached Gemini model or discover one."""
    global _gemini_model, _gemini_model_name

    if _gemini_model is None:
        discover_gemini_model()

    return _gemini_model, _gemini_model_name


def reset_gemini_model():
    """Reset the cached model (useful for retry logic)."""
    global _gemini_model, _gemini_model_name
    _gemini_model = None
    _gemini_model_name = None

# Initialize FastAPI
app = FastAPI(
    title="Physical AI RAG Chatbot",
    description="RAG chatbot powered by Gemini AI for Physical AI documentation",
    version="1.0.0"
)

# CORS for Docusaurus frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic Models ====================

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question")
    context_text: Optional[str] = Field(None, description="User-selected text as context")


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    context_used: str


class IngestRequest(BaseModel):
    content: Optional[str] = Field(None, description="Markdown content to ingest")
    source: Optional[str] = Field(None, description="Source filename")
    ingest_docs_folder: Optional[bool] = Field(False, description="Ingest all docs from folder")


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_created: int
    sources: List[str]


# ==================== Helper Functions ====================

def ensure_collection():
    """Create Qdrant collection if it doesn't exist."""
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE)
        )
        print(f"Created collection: {COLLECTION_NAME}")
    return True


def generate_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    """Generate embedding using Gemini embedding-001."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type
    )
    return result['embedding']


def chunk_text(content: str, source: str) -> List[dict]:
    """Split content into overlapping chunks."""
    chunks = []

    # Clean content
    content = content.strip()
    if not content:
        return chunks

    # Split by headers to preserve structure
    sections = re.split(r'(^#{1,3}\s+.+$)', content, flags=re.MULTILINE)

    current_header = ""
    current_text = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if re.match(r'^#{1,3}\s+', section):
            # Process previous section
            if current_text:
                chunks.extend(create_chunks(current_text, current_header, source))
            current_header = section
            current_text = ""
        else:
            current_text += "\n\n" + section if current_text else section

    # Process last section
    if current_text:
        chunks.extend(create_chunks(current_text, current_header, source))

    return chunks


def create_chunks(text: str, header: str, source: str) -> List[dict]:
    """Create overlapping chunks from text."""
    chunks = []
    text = text.strip()

    if not text:
        return chunks

    # Single chunk if small enough
    if len(text) <= CHUNK_SIZE:
        full_text = f"{header}\n\n{text}" if header else text
        chunks.append({
            "id": hashlib.md5(f"{source}:{full_text[:100]}".encode()).hexdigest(),
            "content": full_text,
            "source": source,
            "header": header
        })
        return chunks

    # Split into overlapping chunks
    start = 0
    idx = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # Break at paragraph or sentence boundary
        if end < len(text):
            para = text.rfind("\n\n", start, end)
            if para > start + CHUNK_SIZE // 2:
                end = para
            else:
                sent = text.rfind(". ", start, end)
                if sent > start + CHUNK_SIZE // 2:
                    end = sent + 1

        chunk_text = text[start:end].strip()
        full_text = f"{header}\n\n{chunk_text}" if header else chunk_text

        chunks.append({
            "id": hashlib.md5(f"{source}:{full_text[:100]}:{idx}".encode()).hexdigest(),
            "content": full_text,
            "source": source,
            "header": header
        })

        start = end - CHUNK_OVERLAP if end < len(text) else len(text)
        idx += 1

    return chunks


def search_qdrant(query: str, limit: int = 5) -> List[dict]:
    """Search Qdrant for relevant chunks."""
    query_embedding = generate_embedding(query, task_type="retrieval_query")

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit,
        score_threshold=0.5
    )

    return [
        {
            "content": hit.payload.get("content", ""),
            "source": hit.payload.get("source", ""),
            "header": hit.payload.get("header", ""),
            "score": hit.score
        }
        for hit in results
    ]


def generate_answer(query: str, context: str, is_selected_context: bool = False, use_general_knowledge: bool = False) -> str:
    """Generate answer using Gemini with dynamic model discovery."""

    if is_selected_context:
        prompt = f"""You are an expert assistant for Physical AI and Humanoid Robotics documentation.
The user has selected specific text and wants you to explain or answer questions about it.

## Selected Text
{context}

## User's Question
{query}

Provide a clear, helpful explanation focused on the selected text. Be concise but thorough."""

    elif use_general_knowledge:
        prompt = f"""You are an expert assistant specializing in Physical AI and Humanoid Robotics.

The user is asking about Physical AI topics, but no specific documentation was found in the knowledge base.
Please answer based on your general knowledge about:
- Physical AI (AI systems that interact with the physical world)
- Humanoid robotics (bipedal robots, locomotion, manipulation)
- Robot perception (computer vision, sensors, SLAM)
- Robot learning (reinforcement learning, imitation learning)
- Control systems for robots

## User's Question
{query}

Provide a helpful, accurate answer based on your knowledge of these topics. Be informative but acknowledge if you're uncertain about specific details."""

    else:
        prompt = f"""You are an expert assistant for Physical AI and Humanoid Robotics documentation.
Answer the user's question based on the provided documentation context.

## Documentation Context
{context}

## User's Question
{query}

Guidelines:
- Base your answer on the provided context
- Be accurate and helpful
- If the context doesn't contain relevant information, say so clearly
- Use technical language appropriate for robotics and AI topics

Provide a helpful answer:"""

    # Get the discovered model
    try:
        model, model_name = get_gemini_model()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[GEMINI] Error generating answer: {e}")
        # Reset and retry once
        reset_gemini_model()
        try:
            model, model_name = get_gemini_model()
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            raise RuntimeError(f"Failed to generate answer after retry: {e2}")


def get_docs_folder() -> Path:
    """Get the docs folder path."""
    # Try relative to backend folder
    backend_dir = Path(__file__).parent
    docs_path = backend_dir.parent / "docs"

    if docs_path.exists():
        return docs_path

    # Fallback
    return Path("../docs")


def read_markdown_files(docs_path: Path) -> List[dict]:
    """Read all markdown files from docs folder."""
    files = []

    if not docs_path.exists():
        return files

    for md_file in docs_path.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            relative_path = md_file.relative_to(docs_path)
            files.append({
                "content": content,
                "source": str(relative_path)
            })
        except Exception as e:
            print(f"Error reading {md_file}: {e}")

    # Also check for .mdx files (Docusaurus)
    for mdx_file in docs_path.rglob("*.mdx"):
        try:
            content = mdx_file.read_text(encoding="utf-8")
            relative_path = mdx_file.relative_to(docs_path)
            files.append({
                "content": content,
                "source": str(relative_path)
            })
        except Exception as e:
            print(f"Error reading {mdx_file}: {e}")

    return files


# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    print("\n" + "=" * 60)
    print("   Physical AI RAG Backend - Starting...")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Gemini API Key: {'[SET]' if GEMINI_API_KEY else '[NOT SET]'}")
    print(f"  Qdrant URL: {QDRANT_URL[:50]}..." if QDRANT_URL and len(QDRANT_URL) > 50 else f"  Qdrant URL: {QDRANT_URL}")

    # Discover and test Gemini model
    if GEMINI_API_KEY:
        try:
            model, model_name = discover_gemini_model()
            print(f"\n  Gemini Model: {model_name}")
        except Exception as e:
            print(f"\n  [WARNING] Gemini initialization failed: {e}")
    else:
        print("\n  [WARNING] GEMINI_API_KEY not set!")

    # Initialize Qdrant
    try:
        ensure_collection()
        print(f"  Qdrant Collection: {COLLECTION_NAME} [READY]")
    except Exception as e:
        print(f"  [WARNING] Qdrant initialization failed: {e}")

    print("\n" + "=" * 60)
    print("   Backend ready!")
    print("   API Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Physical AI RAG Chatbot",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "ingest": "POST /ingest",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health():
    """Health check."""
    qdrant_status = "unknown"
    try:
        qdrant.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)[:50]}"

    return {
        "status": "healthy" if qdrant_status == "connected" else "degraded",
        "gemini": "configured" if GEMINI_API_KEY else "not configured",
        "qdrant": qdrant_status
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG support.

    - If context_text provided: uses it directly as context
    - Otherwise: searches Qdrant for relevant documentation
    - If no results: uses Gemini's general knowledge about Physical AI
    """
    # Debug logging
    print(f"[CHAT] Query: {request.query}")
    if request.context_text:
        print(f"[CHAT] Context provided: {len(request.context_text)} chars")

    try:
        sources = []
        context_used = "none"
        context = ""
        use_general_knowledge = False

        # Determine context
        if request.context_text and request.context_text.strip():
            # Use provided context (user selected text)
            context = request.context_text
            context_used = "selected"
            print("[CHAT] Using selected text as context")

            # Optionally search for additional sources
            try:
                search_results = search_qdrant(request.query, limit=2)
                sources = [{"source": r["source"], "score": r["score"]} for r in search_results]
            except Exception as e:
                print(f"[CHAT] Additional search failed: {e}")
        else:
            # Search Qdrant for context
            try:
                search_results = search_qdrant(request.query, limit=5)
                print(f"[CHAT] Qdrant returned {len(search_results)} results")

                if search_results:
                    context = "\n\n---\n\n".join([
                        f"**{r['source']}**\n{r['content']}"
                        for r in search_results
                    ])
                    sources = [{"source": r["source"], "score": r["score"]} for r in search_results]
                    context_used = "retrieved"
                else:
                    # No results - use general knowledge
                    print("[CHAT] No results, using general knowledge")
                    use_general_knowledge = True
                    context_used = "general"
            except Exception as e:
                # Qdrant error - fallback to general knowledge
                print(f"[CHAT] Qdrant error: {e}, using general knowledge")
                use_general_knowledge = True
                context_used = "general"

        # Generate answer
        print(f"[CHAT] Generating answer (context_used={context_used})")
        answer = generate_answer(
            query=request.query,
            context=context,
            is_selected_context=(context_used == "selected"),
            use_general_knowledge=use_general_knowledge
        )
        print("[CHAT] Answer generated successfully")

        return ChatResponse(
            answer=answer,
            sources=sources,
            context_used=context_used
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest documentation into vector store.

    - If content provided: ingest single document
    - If ingest_docs_folder=True: ingest all docs from docs folder
    """
    try:
        ensure_collection()

        all_chunks = []
        sources_processed = []

        if request.ingest_docs_folder:
            # Ingest all docs from folder
            docs_path = get_docs_folder()
            files = read_markdown_files(docs_path)

            for file in files:
                chunks = chunk_text(file["content"], file["source"])
                all_chunks.extend(chunks)
                sources_processed.append(file["source"])

        elif request.content and request.source:
            # Ingest single document
            chunks = chunk_text(request.content, request.source)
            all_chunks.extend(chunks)
            sources_processed.append(request.source)
        else:
            return IngestResponse(
                status="error",
                message="Provide either content+source or set ingest_docs_folder=True",
                chunks_created=0,
                sources=[]
            )

        if not all_chunks:
            return IngestResponse(
                status="warning",
                message="No content to ingest",
                chunks_created=0,
                sources=sources_processed
            )

        # Generate embeddings and upsert
        points = []
        for chunk in all_chunks:
            embedding = generate_embedding(chunk["content"])
            points.append(
                PointStruct(
                    id=chunk["id"],
                    vector=embedding,
                    payload={
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "header": chunk["header"]
                    }
                )
            )

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

        return IngestResponse(
            status="success",
            message=f"Ingested {len(all_chunks)} chunks from {len(sources_processed)} files",
            chunks_created=len(all_chunks),
            sources=sources_processed
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/collection/info")
async def collection_info():
    """Get collection statistics."""
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection/clear")
async def clear_collection():
    """Clear all vectors from collection."""
    try:
        qdrant.delete_collection(COLLECTION_NAME)
        ensure_collection()
        return {"status": "success", "message": "Collection cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
