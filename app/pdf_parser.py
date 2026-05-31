"""
pdf_parser.py
-------------
Handles all PDF reading and text extraction.

We use 'pdfplumber' which is beginner-friendly and reliable.
It handles:
  - Multi-page PDFs
  - Tables (converts to text)
  - Mixed layouts

The output is clean, plain text ready for NLP processing.
"""

import re
import logging
from pathlib import Path
import pdfplumber

# Create a logger for this module
# Log messages will look like: "pdf_parser - INFO - Processing PDF..."
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    This function:
      1. Validates the file exists and isn't too large
      2. Opens the PDF with pdfplumber
      3. Extracts text from every page
      4. Cleans up the raw text
      5. Returns a single clean string

    Args:
        file_path: Path to the PDF file on disk

    Returns:
        Cleaned text content as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is too large, empty, or unreadable
    """
    path = Path(file_path)

    # ── Validation ──────────────────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Check file size (reject files over 10MB to prevent abuse)
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 10.0:
        raise ValueError(
            f"File too large: {file_size_mb:.1f}MB. Maximum allowed size is 10MB."
        )

    # ── Extraction ──────────────────────────────────────────────────────────
    extracted_pages = []

    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Opening PDF: {path.name} ({total_pages} pages)")

            for page_num, page in enumerate(pdf.pages, start=1):
                # pdfplumber's extract_text() returns None if no text found
                page_text = page.extract_text()

                if page_text:
                    extracted_pages.append(page_text)
                    logger.debug(f"  Page {page_num}/{total_pages}: {len(page_text)} chars")
                else:
                    # This usually happens with scanned (image-based) PDFs
                    logger.warning(
                        f"  Page {page_num}/{total_pages}: no text detected "
                        f"(possible scanned image)"
                    )

    except pdfplumber.pdfminer.high_level.PDFSyntaxError:
        raise ValueError("Corrupted or invalid PDF file.")
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")

    # ── Post-processing ──────────────────────────────────────────────────────
    # Join all pages with a separator
    raw_text = "\n\n--- PAGE BREAK ---\n\n".join(extracted_pages)

    if not raw_text.strip():
        raise ValueError(
            "No text could be extracted. "
            "The PDF may be scanned (image-only). "
            "Please use a text-based PDF."
        )

    # Clean the raw text
    clean = _clean_text(raw_text)
    logger.info(
        f"Extraction complete: {len(extracted_pages)} pages, "
        f"{len(clean)} characters after cleaning"
    )
    return clean


def _clean_text(text: str) -> str:
    """
    Internal helper: Remove noise from extracted PDF text.

    Common PDF extraction issues we fix here:
      - Multiple spaces/newlines left by column layouts
      - Bullet point symbols converted from font glyphs
      - Non-ASCII garbage characters from encoding issues
      - Leading/trailing whitespace

    Args:
        text: Raw text from pdfplumber

    Returns:
        Cleaned text string
    """
    # Replace decorative bullet symbols with plain dashes
    text = re.sub(r'[•·▪▸►◆◇■□●○‣⁃]', '-', text)

    # Remove non-ASCII characters (encoding artifacts like \x93, \x94)
    # Keep standard printable ASCII + basic punctuation
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

    # Normalize whitespace:
    #   Multiple spaces → single space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    #   More than 2 consecutive newlines → 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove page break markers we added (for final output)
    text = text.replace('--- PAGE BREAK ---', '')

    return text.strip()


def extract_text_from_string(raw_text: str) -> str:
    """
    For when the user pastes raw text instead of uploading a PDF.
    Just validates and cleans the input.

    Args:
        raw_text: User-pasted text

    Returns:
        Cleaned text

    Raises:
        ValueError: If text is empty
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Text input is empty.")

    return _clean_text(raw_text)
