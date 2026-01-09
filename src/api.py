"""FastAPI application for the CV Q&A Agent service.

Exposes HTTP endpoints to query the agent graph with questions about a CV.
The agent responds using multiple personalities (default, mysterious, etc.).
"""

import os
import sys
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


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class QuestionRequest(BaseModel):
    """Request model for asking a question to the agent."""

    question: str
    include_history: Optional[bool] = False


class PersonalityAnswer(BaseModel):
    """Single answer from a personality."""

    personality: str
    answer: str


class QuestionResponse(BaseModel):
    """Response model containing answers from all personalities."""

    question: str
    answers: dict[str, str]  # personality -> answer
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


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the CV.

    The agent will respond with answers from multiple personalities
    (default, mysterious, etc.) in parallel.

    Args:
        request: QuestionRequest containing the question text

    Returns:
        QuestionResponse with answers grouped by personality

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
    logger.info(f"Processing question: {question}")

    try:
        # Prepare the state for the graph
        state = {
            "cv_text": cv_text,
            "question": question,
            "history": [],
            "responses": [],
        }

        # Invoke the compiled graph
        result = compiled_graph.invoke(state)

        # Extract and format the answers
        answers_dict = {}
        for response in result.get("responses", []):
            # Each response is a dict with personality as key
            answers_dict.update(response)

        logger.info(f"Got {len(answers_dict)} personality responses")

        return QuestionResponse(
            question=question,
            answers=answers_dict,
            success=True,
        )

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
