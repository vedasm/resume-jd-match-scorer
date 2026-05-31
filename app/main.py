"""
main.py
-------
The FastAPI application — the heart of the project.

WHAT FASTAPI DOES:
  - Handles HTTP requests (GET, POST)
  - Validates uploaded files
  - Calls our NLP and PDF modules
  - Returns JSON responses

ROUTES (URL endpoints):
  GET  /            → Show the web UI (index.html)
  POST /analyze     → Accept two PDFs, return analysis JSON
  POST /analyze-text→ Same but accepts plain text (for testing)
  GET  /health      → Simple health check (used by Docker)
"""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from dotenv import load_dotenv

# Import our modules
from app.gap_analysis import run_gap_analysis
from app.models import AnalysisResult
from app.nlp_utils import load_model, run_full_analysis
from app.pdf_parser import extract_text_from_pdf, extract_text_from_string

# Load .env environment variables
load_dotenv()

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume–JD Match Scorer",
    description="Semantic resume analysis using Sentence-Transformers",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI available at /docs
    redoc_url="/redoc",     # Alternative docs at /redoc
)

# Crucial for Hugging Face Spaces Load Balancers
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Enable CORS (Allows the iframe to talk to the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Template engine ────────────────────────────────────────────────────────
# Jinja2 renders our HTML template with dynamic data
templates = Jinja2Templates(directory="app/templates")

# Allowed MIME types for uploaded files
ALLOWED_TYPES = {"application/pdf", "application/octet-stream"}


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def warmup():
    """
    Pre-load the NLP model when the server starts.

    WHY: The first time generate_embedding() is called, it loads the model
    (~80MB) from disk. This takes 5–15 seconds. By pre-loading at startup,
    the first user request is just as fast as subsequent ones.
    """
    logger.info("Warming up NLP model (this takes a moment on first run)...")
    load_model()   # @lru_cache ensures subsequent calls are instant
    logger.info("Server is ready. Visit http://localhost:8000")


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Serve the main web interface.
    When you open http://localhost:8000 in a browser, you see this page.
    """
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/analyze")
async def analyze_pdfs(
    resume: UploadFile = File(..., description="Resume PDF"),
    job_description: UploadFile = File(..., description="Job description PDF"),
):
    """
    Core analysis endpoint.

    WORKFLOW:
      1. Validate uploaded files (type check)
      2. Save to temporary files on disk
      3. Extract text from PDFs
      4. Compute semantic similarity (NLP)
      5. Run gap analysis
      6. Return JSON result
      7. Clean up temp files

    The 'finally' block ensures temp files are ALWAYS deleted, even on errors.

    Args:
        resume:          UploadFile — the candidate's resume PDF
        job_description: UploadFile — the target job description PDF

    Returns:
        JSON with AnalysisResult structure
    """
    # File validation
    for uf, label in [(resume, "resume"), (job_description, "job description")]:
        if uf.content_type not in ALLOWED_TYPES and not uf.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"'{label}' must be a PDF file. Got: {uf.content_type}"
            )

    resume_path = jd_path = None  # Will be set before try block needs them

    try:
        # ── Save uploads to temp files ─────────────────────────────────────
        # tempfile.NamedTemporaryFile creates a file that auto-deletes
        # We use delete=False because we delete manually in 'finally'
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(await resume.read())
            resume_path = f.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(await job_description.read())
            jd_path = f.name

        logger.info(f"Processing: '{resume.filename}' vs '{job_description.filename}'")

        # ── Step 1: Extract text ───────────────────────────────────────────
        logger.info("Step 1/3 — Extracting text from PDFs...")
        resume_text = extract_text_from_pdf(resume_path)
        jd_text     = extract_text_from_pdf(jd_path)

        # ── Step 2: NLP analysis ───────────────────────────────────────────
        logger.info("Step 2/3 — Computing semantic similarity...")
        nlp_result = run_full_analysis(resume_text, jd_text)

        # ── Step 3: Gap analysis ───────────────────────────────────────────
        logger.info("Step 3/3 — Running gap analysis...")
        gap_result = run_gap_analysis(resume_text, jd_text)

        # ── Assemble result ────────────────────────────────────────────────
        result = AnalysisResult(
            similarity_score=nlp_result["similarity_score"],
            score_percentage=f"{nlp_result['similarity_score']:.1f}%",
            skill_gaps=gap_result["skill_gaps"],
            resume_skills=gap_result["resume_skills"],
            jd_skills=gap_result["jd_skills"],
            matching_skills=gap_result["matching_skills"],
            processing_time=nlp_result["processing_time"],
        )

        logger.info(f"Done → score={result.score_percentage}, gaps={len(result.skill_gaps)}")
        return result.model_dump()

    except ValueError as e:
        # User errors (bad PDF, empty file, wrong type)
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(status_code=500, detail="Internal error. Please try again.")

    finally:
        # ── Always clean up temp files ─────────────────────────────────────
        for path in [resume_path, jd_path]:
            if path and Path(path).exists():
                Path(path).unlink()


@app.post("/analyze-text")
async def analyze_plain_text(
    resume_text: str = Form(...),
    jd_text: str = Form(...),
):
    """
    Alternative endpoint accepting plain text instead of PDFs.
    Useful for testing without needing PDF files.

    Example (curl):
        curl -X POST http://localhost:8000/analyze-text \\
             -F "resume_text=Python developer with 3 years experience in Django" \\
             -F "jd_text=Seeking Python engineer with Django and PostgreSQL experience"
    """
    try:
        resume_clean = extract_text_from_string(resume_text)
        jd_clean     = extract_text_from_string(jd_text)

        nlp_result = run_full_analysis(resume_clean, jd_clean)
        gap_result = run_gap_analysis(resume_clean, jd_clean)

        result = AnalysisResult(
            similarity_score=nlp_result["similarity_score"],
            score_percentage=f"{nlp_result['similarity_score']:.1f}%",
            skill_gaps=gap_result["skill_gaps"],
            resume_skills=gap_result["resume_skills"],
            jd_skills=gap_result["jd_skills"],
            matching_skills=gap_result["matching_skills"],
            processing_time=nlp_result["processing_time"],
        )
        return result.model_dump()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error in /analyze-text")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """
    Docker health check endpoint.
    Returns 200 OK when the server is running.
    """
    return {"status": "ok", "version": "1.0.0"}
