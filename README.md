# PyDocFlow

**Document processing service with OCR and LLM-powered text extraction and enhancement.**

PyDocFlow is a FastAPI-based backend service that processes documents (PDFs and images), extracts text using OCR (Tesseract), and optionally enhances the output using Large Language Models (LLMs).

---

## Features

- **Multi-format Support**: Process PDFs and images (PNG, JPEG, TIFF, BMP, WebP)
- **OCR Engine**: Tesseract-based OCR with support for multiple languages (default: Arabic)
- **Smart PDF Detection**: Automatically detects scanned vs. native PDFs
- **LLM Integration**: Enhance extracted text using LLMs (Ollama, OpenAI, Anthropic, etc.)
- **Parallel Processing**: Uses ProcessPoolExecutor for efficient OCR batch processing
- **MongoDB Persistence**: Store and retrieve OCR results
- **Docker Ready**: Includes Dockerfile and docker-compose.yml

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                      │
├─────────────────────────────────────────────────────────────────┤
│  Endpoints                                                        │
│  • POST /upload-file      - Single file upload                   │
│  • POST /upload-files     - Multiple file upload                 │
│  • POST /parse-with-llm   - OCR + LLM enhancement                │
│  • POST /enhance-text     - Text enhancement only                │
│  • GET  /health           - Health check                         │
├─────────────────────────────────────────────────────────────────┤
│  Services                                                         │
│  • IngestionService  - Document intake & preparation             │
│  • ParsingService    - OCR parsing & LLM enhancement             │
├─────────────────────────────────────────────────────────────────┤
│  Core Components                                                  │
│  • OCREngine         - Tesseract OCR processing                  │
│  • LLMProvider       - LLM inference (Ollama/LiteLLM)            │
├─────────────────────────────────────────────────────────────────┤
│  Persistence                                                      │
│  • MongoDB (Motor/ODMantic)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.12+
- Tesseract OCR installed on your system
- (Optional) Ollama for local LLM inference
- (Optional) MongoDB for persistence

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PyDocFlow
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. Install Tesseract

**Windows:**
```powershell
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-ara
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 4. Install Ollama (Optional)

```bash
# Visit https://ollama.ai for installation
# Then pull a model:
ollama pull llama3.2
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Environment Variables

```ini
# Database Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=pydocflow

# OCR Configuration
OCR_LANGUAGE=ara

# LLM Configuration
LLM_PROVIDER=ollama           # Options: ollama, litellm, mock
LLM_MODEL=llama3.2            # Model name
LLM_OLLAMA_HOST=http://localhost:11434

# Generation Parameters
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
LLM_TIMEOUT=120
```

### LLM Provider Examples

**Ollama (Local - Default):**
```ini
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_OLLAMA_HOST=http://localhost:11434
```

**OpenAI via LiteLLM:**
```ini
LLM_PROVIDER=litellm
LLM_MODEL=openai/gpt-4
LLM_API_KEY=sk-your-api-key
```

**Anthropic via LiteLLM:**
```ini
LLM_PROVIDER=litellm
LLM_MODEL=anthropic/claude-3-sonnet-20240229
LLM_API_KEY=your-anthropic-key
```

**Mock (Testing):**
```ini
LLM_PROVIDER=mock
```

---

## Usage

### Run Docker Compose (with MongoDB)

```bash
docker-compose up -d
```

### Start the Server

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker-compose up --build
```

### API Endpoints

#### 1. Upload Single File

```bash
curl -X POST "http://localhost:8000/upload-file" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "results": ["extracted text from page 1", "extracted text from page 2"]
}
```

#### 2. Upload Multiple Files

```bash
curl -X POST "http://localhost:8000/upload-files" \
  -F "files=@doc1.pdf" \
  -F "files=@image.png"
```

**Response:**
```json
{
  "results": [
    {"filename": "doc1.pdf", "content": ["page 1 text", "page 2 text"]},
    {"filename": "image.png", "content": ["extracted text"]}
  ]
}
```

#### 3. Parse with LLM Enhancement

```bash
curl -X POST "http://localhost:8000/parse-with-llm" \
  -F "file=@document.pdf" \
  -F "prompt=Summarize the key points"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "content": "LLM-enhanced summary...",
  "llm_model": "ollama"
}
```

#### 4. Enhance Existing Text

```bash
curl -X POST "http://localhost:8000/enhance-text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Raw extracted text...",
    "instruction": "Translate to Arabic"
  }'
```

#### 5. Health Check

```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
  "healthy": true,
  "status": "healthy",
  "components": {
    "database": {"healthy": true},
    "ingestion_service": {"healthy": true},
    "parsing_service": {"healthy": true},
    "llm_provider": {"healthy": true}
  }
}
```

---

## Project Structure

```
PyDocFlow/
├── main.py                 # FastAPI application entry point
├── pyproject.toml          # Project dependencies and metadata
├── docker-compose.yml      # Docker services orchestration
├── Dockerfile              # Container build instructions
├── .env.example            # Environment configuration template
│
├── llm/                    # LLM integration module
│   ├── __init__.py
│   ├── base.py             # LLMProvider abstract base class
│   ├── config.py           # LLMConfig settings model
│   ├── errors.py           # Custom exceptions
│   ├── registry.py         # Provider factory/registry
│   └── providers/
│       ├── ollama.py       # Ollama provider
│       ├── litellm.py      # LiteLLM provider (OpenAI, Anthropic, etc.)
│       └── mock.py         # Mock provider for testing
│
├── ocr/                    # OCR module
│   ├── __init__.py
│   └── pytesseract_engine.py  # Tesseract OCR engine
│
├── services/               # Business logic services
│   ├── __init__.py
│   ├── base.py             # BaseService abstract class
│   ├── ingestion.py        # Document ingestion service
│   └── parsing.py          # Parsing & LLM enhancement service
│
├── persistence/            # Database layer
│   ├── __init__.py
│   ├── db.py               # Database connection manager
│   ├── models.py           # ODMantic models
│   └── repository.py       # Data access repository
│
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_base_service.py
│   ├── test_ingestion.py
│   ├── test_llm.py
│   └── ...
│
└── data/                   # Test data directory
    ├── *.pdf               # Sample PDFs
    └── *.png               # Sample images
```

---

## Development

### Run Tests

```bash
uv run pytest tests/ -v
```

### Code Style

This project follows Python best practices with type hints and async/await patterns.

### Adding a New LLM Provider

1. Create a new provider class in `llm/providers/`:

```python
from llm.base import LLMProvider, LLMResponse

class MyProvider(LLMProvider):
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        # Implementation
        pass

    async def chat(self, messages: list, **kwargs) -> LLMResponse:
        # Implementation
        pass

    async def health_check(self) -> dict:
        # Implementation
        pass
```

2. Register in `llm/registry.py`:

```python
LLMRegistry.register("myprovider", MyProvider)
```

---

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Troubleshooting

### Tesseract Not Found

Ensure Tesseract is installed and in your PATH:

```bash
tesseract --version
```

### Ollama Connection Error

Check that Ollama is running:

```bash
ollama list
```

### MongoDB Connection Failed

Verify MongoDB is running:

```bash
mongod --version
```

---

## License

This project is proprietary software. All rights reserved.

---

## Author

**M.S.KADRI**  
Email: [ms.kadri.dev@gmail.com](mailto:ms.kadri.dev@gmail.com)

---

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Ollama](https://ollama.ai/)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [Motor](https://motor.readthedocs.io/)
- [ODMantic](https://art049.github.io/odmantic/)
