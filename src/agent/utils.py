"""Utility & helper functions."""

import logging
import os
from typing import Dict

import structlog
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Load environment variables from a .env file
load_dotenv()


# Configure the standard Python logger to use RichHandler for colored output
custom_theme = Theme(
    {
        "info": "green",
        "warning": "yellow",
        "error": "bold red",
        "debug": "cyan",
    }
)

# Rich console used for manual prints from agents
CONSOLE = Console(theme=custom_theme)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            show_time=True,
            log_time_format="%H:%M:%S",
            show_level=True,
            show_path=True,
            enable_link_path=True,
            console=CONSOLE,
        )
    ],
)

# Silenzia i log di librerie esterne
for noisy in ["httpcore", "httpx", "urllib3", "asyncio", "selector_events", "models"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# Configure structlog for structured logging
structlog.configure(
    processors=[
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.KeyValueRenderer(key_order=["event"]),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


def create_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Create and return a structured logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        structlog.stdlib.BoundLogger: The configured structured logger.
    """
    return structlog.get_logger(name)


def get_llm_instance(t: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Configure and return an instance of the LLM model with specific parameters.
    Also checks for rate limit issues by making a test call.

    Args:
        t (float, optional): Temperature setting for the model. Defaults to 0.0.

    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance.
    """
    model_name = os.getenv("GEMINI_MODEL", "GEMINI_MODEL_BACKOFF")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=t,
        max_retries=2,
        google_api_key=google_api_key,
    )
    return llm


if __name__ == "__main__":
    # Test logger
    logger = create_logger("test")
    logger.info("Logger initialized successfully")

    try:
        # Minimal test prompt, adjust as needed
        llm = get_llm_instance()
        resp = llm.invoke("ping")
        logger.info("LLM instance created and test call successful", response=resp)
    except Exception as e:

        logger.error("Failed to instantiate LLM or hit rate limit", error=str(e))
        raise RuntimeError("LLM instantiation failed or rate limit reached") from e
