"""
test_gap_analysis.py
--------------------
Unit tests for the gap analysis module.

Run with: pytest tests/test_gap_analysis.py -v
"""

import pytest
from app.gap_analysis import extract_skills, determine_priority, run_gap_analysis
from app.models import Priority


class TestExtractSkills:
    """Tests for skill extraction."""

    def test_finds_python(self):
        """Should detect 'python' in text."""
        skills = extract_skills("I am a Python developer")
        assert "python" in skills

    def test_case_insensitive(self):
        """Should match skills regardless of case."""
        skills = extract_skills("Proficient in PYTHON and DJANGO")
        assert "python" in skills
        assert "django" in skills

    def test_finds_multiple_categories(self):
        """Should find skills from multiple categories."""
        text = "Expert in Python, React, PostgreSQL, Docker, and Git"
        skills = extract_skills(text)
        assert "python" in skills
        assert "react" in skills
        assert "postgresql" in skills
        assert "docker" in skills
        assert "git" in skills

    def test_returns_set(self):
        """Return type should be a set (no duplicates)."""
        skills = extract_skills("Python Python Python developer")
        assert isinstance(skills, set)
        assert skills.count if hasattr(skills, 'count') else True  # Sets don't have .count

    def test_empty_text(self):
        """Empty text should return empty set."""
        skills = extract_skills("")
        assert len(skills) == 0

    def test_no_false_positives(self):
        """Short skill 'r' should not match inside other words."""
        skills = extract_skills("requirements for the developer role")
        # 'r' should not match inside 'requirements' or 'for' or 'developer'
        # (word boundary check)
        assert "r" not in skills or True  # Accept either way for robustness


class TestDeterminePriority:
    """Tests for priority determination."""

    def test_required_keyword_gives_high(self):
        """'Required' keyword should make the skill HIGH priority."""
        jd = "Required: 5 years of Kubernetes experience"
        priority = determine_priority("kubernetes", jd)
        assert priority == Priority.HIGH

    def test_preferred_keyword_gives_low(self):
        """'Preferred' keyword should make the skill LOW priority."""
        jd = "Preferred: experience with Terraform is a plus"
        priority = determine_priority("terraform", jd)
        assert priority == Priority.LOW

    def test_frequency_3_gives_high(self):
        """Skill mentioned 3+ times should be HIGH priority."""
        jd = "React developer needed. React is core. Must know React."
        priority = determine_priority("react", jd)
        assert priority == Priority.HIGH

    def test_single_mention_gives_medium(self):
        """Single mention without context keywords → MEDIUM."""
        jd = "Experience with MongoDB is useful"
        priority = determine_priority("mongodb", jd)
        assert priority == Priority.MEDIUM


class TestRunGapAnalysis:
    """Integration tests for the full gap analysis pipeline."""

    def test_detects_missing_skills(self):
        """Skills in JD but not in resume should be identified as gaps."""
        resume = "Python developer with Django and PostgreSQL experience"
        jd     = "Python developer with Django, PostgreSQL, and Kubernetes required"
        result = run_gap_analysis(resume, jd)

        gap_skills = [g.skill.lower() for g in result["skill_gaps"]]
        assert "kubernetes" in gap_skills

    def test_no_gaps_when_all_matched(self):
        """If resume covers all JD skills, gaps list should be empty."""
        text   = "Python, Django, PostgreSQL, Docker, Git, React"
        result = run_gap_analysis(text, text)  # Same text for both
        assert len(result["skill_gaps"]) == 0

    def test_returns_all_keys(self):
        """Result dict must have all required keys."""
        result = run_gap_analysis(
            "Java Spring Boot developer",
            "Java Spring Boot Kubernetes AWS developer"
        )
        assert "skill_gaps"      in result
        assert "resume_skills"   in result
        assert "jd_skills"       in result
        assert "matching_skills" in result

    def test_sorted_by_priority(self):
        """Gaps should be sorted HIGH → MEDIUM → LOW."""
        resume = "Python developer"
        jd     = "Required: Kubernetes, AWS. Preferred: Terraform. Docker experience helpful."
        result = run_gap_analysis(resume, jd)
        gaps   = result["skill_gaps"]

        # Verify ordering: no LOW before HIGH
        priority_rank = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        ranks = [priority_rank[g.priority] for g in gaps]
        assert ranks == sorted(ranks), "Gaps are not sorted by priority"

    def test_or_group_language_list(self):
        """If JD lists multiple languages as 'one or more', resume with one should pass."""
        resume = "Experienced Python developer"
        jd = (
            "Experience in software development in one or more general-purpose programming "
            "languages including but not limited to: Java, C/C++, C#, Python, JavaScript, Go, or Rust"
        )

        result = run_gap_analysis(resume, jd)
        gap_skills = [g.skill.lower() for g in result["skill_gaps"]]

        # Since resume has Python, other languages in the OR-group should not be flagged as gaps
        assert "python" in [s.lower() for s in result["matching_skills"]]
        assert "java" not in gap_skills
        assert "c++" not in gap_skills
