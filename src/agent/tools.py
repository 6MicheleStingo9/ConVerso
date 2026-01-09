"""Tools for the agent to use."""

import os
from PyPDF2 import PdfReader
from .utils import create_logger

logger = create_logger("agent.tools")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Text extracted from the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def get_message_content(msg):
    if hasattr(msg, "content"):
        return msg.content
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"]
    return str(msg)


if __name__ == "__main__":
    pdf_path = os.path.join(os.path.dirname(__file__), "../files/ms_cv.pdf")
    try:
        text = extract_text_from_pdf(pdf_path)
        print(text)
    except Exception as e:
        logger.error(f"Error: {e}")
