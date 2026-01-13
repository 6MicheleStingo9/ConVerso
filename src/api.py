"""FastAPI application for the CV Q&A Agent service.

Exposes HTTP endpoints to query the agent graph with questions about a CV.
The agent responds using multiple personalities (default, mysterious, etc.).
"""

import os
import sys
import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import build_agent_graph
from agent.tools import extract_text_from_pdf
from agent.utils import create_logger

logger = create_logger("api")

# ============================================================================
# GLOBAL STATE
# ============================================================================

# These will be initialized in the lifespan event
compiled_graph = None
cv_text = None

# Session store: session_id -> {"history": [...], "last_access": timestamp}
sessions: dict[str, dict] = {}
SESSION_TTL_SECONDS = 1800  # 30 minutes
MAX_HISTORY_ENTRIES = 20  # Limit history size per session


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class QuestionRequest(BaseModel):
    """Request model for asking a question to the agent."""

    question: str
    session_id: Optional[str] = None


class PersonalityAnswer(BaseModel):
    """Single answer from a personality."""

    personality: str
    answer: str


class QuestionResponse(BaseModel):
    """Response model containing answers from all personalities."""

    question: str
    answers: dict[str, str]  # personality -> answer
    session_id: str
    history_length: int
    success: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    cv_loaded: bool
    graph_initialized: bool


# ============================================================================
# LIFESPAN EVENT
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    global compiled_graph, cv_text

    logger.info("Starting up API server...")

    try:
        # Load the CV text from file
        cv_path = os.path.join(os.path.dirname(__file__), "files/ms_cv.pdf")

        if not os.path.exists(cv_path):
            raise FileNotFoundError(f"CV file not found at {cv_path}")

        logger.info(f"Loading CV from {cv_path}")
        cv_text = extract_text_from_pdf(cv_path)
        logger.info(f"CV loaded successfully ({len(cv_text)} characters)")

        # Build and compile the graph
        logger.info("Building agent graph...")
        graph = build_agent_graph()
        compiled_graph = graph.compile()
        logger.info("Agent graph compiled successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")
        raise

    yield

    # Cleanup (if needed)
    logger.info("Shutting down API server...")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="CV Q&A Agent API",
    description="Multi-personality Q&A service for CV analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and agent are ready."""
    return HealthResponse(
        status="healthy" if compiled_graph and cv_text else "degraded",
        cv_loaded=cv_text is not None,
        graph_initialized=compiled_graph is not None,
    )


def cleanup_expired_sessions():
    """Remove sessions that have exceeded TTL."""
    now = time.time()
    expired = [
        sid
        for sid, data in sessions.items()
        if now - data["last_access"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del sessions[sid]
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")


def get_or_create_session(session_id: Optional[str]) -> tuple[str, list]:
    """Get existing session or create a new one.

    Returns:
        Tuple of (session_id, history)
    """
    cleanup_expired_sessions()

    if session_id and session_id in sessions:
        sessions[session_id]["last_access"] = time.time()
        return session_id, sessions[session_id]["history"]

    new_id = str(uuid.uuid4())
    sessions[new_id] = {"history": [], "last_access": time.time()}
    logger.info(f"Created new session: {new_id}")
    return new_id, []


def update_session_history(session_id: str, new_history: list):
    """Update session history, enforcing max entries limit."""
    if session_id in sessions:
        # Keep only the last MAX_HISTORY_ENTRIES
        sessions[session_id]["history"] = new_history[-MAX_HISTORY_ENTRIES:]
        sessions[session_id]["last_access"] = time.time()


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the CV.

    The agent will respond with answers from multiple personalities
    (default, mysterious, etc.) in parallel. Session history is maintained
    server-side using the session_id.

    Args:
        request: QuestionRequest containing the question and optional session_id

    Returns:
        QuestionResponse with answers grouped by personality and session_id

    Raises:
        HTTPException: If the CV is not loaded or question is empty
    """
    global compiled_graph, cv_text

    # Validate prerequisites
    if not cv_text:
        logger.error("CV text not loaded")
        raise HTTPException(status_code=503, detail="CV not loaded")

    if not compiled_graph:
        logger.error("Graph not compiled")
        raise HTTPException(status_code=503, detail="Agent not ready")

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    question = request.question.strip()

    # Get or create session
    session_id, history = get_or_create_session(request.session_id)
    logger.info(
        f"Processing question for session {session_id[:8]}... (history: {len(history)} entries)"
    )

    try:
        # Prepare the state for the graph with existing history
        state = {
            "cv_text": cv_text,
            "question": question,
            "history": history,
            "responses": [],
        }

        # Invoke the compiled graph
        result = compiled_graph.invoke(state)

        # Extract and format the answers
        answers_dict = {}
        for response in result.get("responses", []):
            answers_dict.update(response)

        # Update session with new history from graph
        new_history = result.get("history", [])
        update_session_history(session_id, new_history)

        logger.info(
            f"Got {len(answers_dict)} personality responses, history now {len(new_history)} entries"
        )

        resp = QuestionResponse(
            question=question,
            answers=answers_dict,
            session_id=session_id,
            history_length=len(new_history),
            success=True,
        )

        # Log the full response payload for debugging
        logger.info(f"/ask response payload: {resp.model_dump_json()}")

        return resp

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CV Q&A Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /ask": "Ask a question about the CV",
            "GET /health": "Health check",
        },
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
    )
