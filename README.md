# Resume–JD Match Scorer

An NLP-powered tool that compares a resume against a job description,
returning a semantic similarity score and a prioritized skill gap analysis.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Web framework | FastAPI |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| PDF parsing | pdfplumber |
| Similarity | scikit-learn cosine_similarity |
| Containerization | Docker + docker-compose |

## Quick Start (Local)

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone / download the project
cd resume-matcher

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app.main:app --reload --port 8000

# 5. Open in browser
# → http://localhost:8000
```

## Quick Start (Docker)

```bash
# 1. Build and start (first run downloads model, ~2 min)
docker-compose up --build

# 2. Open in browser
# → http://localhost:8000

# 3. Stop
docker-compose down
```

## API Endpoints

| Method | URL | Description |
|---|---|---|
| GET  | `/`            | Web UI |
| POST | `/analyze`     | Upload two PDFs → get analysis |
| POST | `/analyze-text`| Submit plain text → get analysis |
| GET  | `/health`      | Health check |
| GET  | `/docs`        | Swagger API docs |

## Running Tests

```bash
# Install test runner
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_nlp_utils.py -v
```

## Project Architecture

See ARCHITECTURE section in the blueprint document for the full diagram.

---
title: Resume JD Match Scorer
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
