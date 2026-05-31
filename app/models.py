"""
models.py
---------
Defines the data structures used throughout the application.

We use Pydantic models which provide:
  - Automatic type validation (e.g., score must be a float, not a string)
  - Auto-generated API documentation
  - Easy JSON serialization (.model_dump())

Think of these as "blueprints" that define what our data looks like.
"""

from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class Priority(str, Enum):
    """
    Enum for skill gap priority levels.
    Using an Enum ensures we only use valid priority values — no typos!
    
    We inherit from 'str' so it serializes to plain text in JSON:
      Priority.HIGH → "High"  (not "Priority.HIGH")
    """
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


class SkillGap(BaseModel):
    """
    Represents a single missing skill identified during gap analysis.
    
    Example:
        SkillGap(
            skill="Kubernetes",
            priority=Priority.HIGH,
            action_item="Get hands-on via free tier; pursue official certification",
            category="Cloud & DevOps"
        )
    """
    skill:       str       # Name of the missing skill (e.g., "Kubernetes")
    priority:    Priority  # How important this gap is (HIGH / MEDIUM / LOW)
    action_item: str       # What the user should do about it
    category:    str       # Which category the skill belongs to
    free_resource: str     # URL to a free learning resource
    paid_resource: str     # URL to a paid learning resource


class AnalysisResult(BaseModel):
    """
    The complete result returned to the user after analysis.
    
    This is what gets sent as JSON when you call POST /analyze.
    """
    similarity_score: float = Field(
        ...,
        ge=0.0,           # Must be >= 0
        le=100.0,         # Must be <= 100
        description="Semantic similarity score (0–100)"
    )
    score_percentage:  str        # Formatted score string, e.g. "78.5%"
    skill_gaps:        List[SkillGap]  # List of missing skills with actions
    resume_skills:     List[str]  # All skills found in resume
    jd_skills:         List[str]  # All skills found in job description
    matching_skills:   List[str]  # Skills present in BOTH (matches)
    processing_time:   float      # How long analysis took (seconds)
