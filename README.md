# ConVersum - CV Q&A Agent API

A multi-personality AI assistant that answers questions about your CV using Google's Gemini model. The agent responds with different personas to provide diverse perspectives on your professional background.

## Features

- 🤖 **Multi-Personality Responses**: Get answers from different personas (default, mysterious, etc.)
- 📄 **PDF CV Processing**: Automatically extracts and understands your CV content
- 🚀 **RESTful API**: Easy-to-use HTTP endpoints for integration
- 📚 **Interactive Docs**: Swagger UI for testing endpoints directly
- 🔄 **Session History**: Maintains context across multiple questions

## Prerequisites

- Python 3.10+
- Google API Key (Gemini)
- pip or conda

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/ConVersum.git
cd ConVersum
```

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Get your Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

5. **Place your CV**

Add your CV as a PDF file:

```
src/files/your_cv_file.pdf
```

## Running the Server

```bash
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

Check if the API is ready:

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "cv_loaded": true,
  "graph_initialized": true
}
```

### Ask a Question

Submit a question about the CV:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are your main skills?"}'
```

Response:

```json
{
  "question": "What are your main skills?",
  "answers": {
    "default": "Based on your CV...",
    "mysterious": "In the shadows of your experience..."
  },
  "success": true
}
```

### Interactive Documentation

Open your browser and visit:

```
http://localhost:8000/docs
```

This provides an interactive Swagger UI where you can test all endpoints directly.

## Project Structure

```
ConVersum/
├── src/
│   ├── api.py                 # FastAPI application
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py           # LangGraph agent definition
│   │   ├── prompts.py         # System prompts for different personalities
│   │   ├── tools.py           # PDF extraction and utilities
│   │   └── utils.py           # Helper functions
│   └── files/
│       └── ms_cv.pdf          # Your CV (not tracked in git)
├── requirements.txt            # Python dependencies
├── Procfile                    # Deployment configuration
├── .env                        # Environment variables (not tracked in git)
└── README.md
```

## Technologies

- **FastAPI**: Modern Python web framework
- **LangGraph**: State graph framework for multi-step agents
- **LangChain**: LLM orchestration
- **Google Gemini**: AI model
- **PyPDF2**: PDF text extraction

## License

MIT License - see LICENSE file for details
