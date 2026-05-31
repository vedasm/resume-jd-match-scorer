"""
test_pdf_parser.py
------------------
Unit tests for the PDF parsing module.

Run with: pytest tests/test_pdf_parser.py -v
"""

import pytest
import tempfile
import os
from app.pdf_parser import extract_text_from_string, _clean_text

# =============================================================================
# Tests for extract_text_from_string
# =============================================================================

class TestExtractTextFromString:
    """Tests for the string input extraction function."""

    def test_basic_text_returned(self):
        """Should return the input text (cleaned)."""
        result = extract_text_from_string("Python developer with Django experience")
        assert "python developer" in result.lower()

    def test_empty_string_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            extract_text_from_string("")

    def test_whitespace_only_raises(self):
        """Whitespace-only input should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            extract_text_from_string("   \n\t  ")

    def test_unicode_cleaned(self):
        """Non-ASCII characters should be removed."""
        result = extract_text_from_string("Python \x93developer\x94")
        assert "\x93" not in result
        assert "\x94" not in result


# =============================================================================
# Tests for _clean_text (internal helper — we test it anyway for robustness)
# =============================================================================

class TestCleanText:
    """Tests for the text cleaning function."""

    def test_removes_excess_newlines(self):
        """Triple (or more) newlines should become double newlines."""
        result = _clean_text("Line 1\n\n\n\n\nLine 2")
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_removes_excess_spaces(self):
        """Multiple spaces should collapse to one."""
        result = _clean_text("Hello     World")
        assert "Hello World" in result

    def test_replaces_bullet_symbols(self):
        """Common PDF bullet glyphs should become hyphens."""
        result = _clean_text("• Python\n▸ JavaScript\n● TypeScript")
        assert "•" not in result
        assert "▸" not in result
        assert "●" not in result

    def test_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        result = _clean_text("   hello world   ")
        assert result == result.strip()

    def test_preserves_content(self):
        """Cleaning should not remove actual content."""
        original = "5 years experience in Python and machine learning"
        result = _clean_text(original)
        assert "Python" in result
        assert "machine learning" in result


# =============================================================================
# Tests for file validation
# =============================================================================

class TestFileSizeValidation:
    """Tests for file-level validation in extract_text_from_pdf."""

    def test_nonexistent_file_raises(self):
        """Should raise FileNotFoundError for missing files."""
        from app.pdf_parser import extract_text_from_pdf
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("/nonexistent/path/file.pdf")

    def test_file_too_large_raises(self):
        """Should raise ValueError for files over 10MB."""
        import io
        from app.pdf_parser import extract_text_from_pdf

        # Create a temp file that's "too large" (we mock the stat)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake" * 100)   # small file
            tmp_path = f.name

        try:
            # Monkey-patch stat to return large size
            import unittest.mock as mock
            mock_stat = mock.MagicMock()
            mock_stat.st_size = 11 * 1024 * 1024  # 11MB

            with mock.patch("pathlib.Path.stat", return_value=mock_stat):
                with pytest.raises(ValueError, match="too large"):
                    extract_text_from_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)
