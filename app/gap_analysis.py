"""
gap_analysis.py
---------------
Identifies skill gaps between a resume and job description.

APPROACH:
  We maintain a hand-curated database of ~120 industry skills across 7 categories.
  For each text, we find which skills are mentioned using regex.
  Skills in JD but NOT in resume = gaps.
  
  This is intentionally simple — for a research paper, you can extend it with:
    - spaCy NER for richer entity extraction
    - TF-IDF to weight skills by document importance
    - A domain-specific skill ontology (e.g., ESCO, O*NET)
"""

import re
import logging
import urllib.parse
from typing import Dict, List, Set

from app.models import Priority, SkillGap

logger = logging.getLogger(__name__)


# =============================================================================
# SKILLS DATABASE
# Organized by category. Each list item is a skill name (lowercase).
# We keep them lowercase to simplify matching.
# =============================================================================

SKILLS_DB: Dict[str, List[str]] = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "golang", "rust", "kotlin", "swift", "scala", "r", "matlab",
        "perl", "ruby", "php", "dart", "julia", "bash", "shell"
    ],
    "Web Frameworks": [
        "react", "angular", "vue", "svelte", "next.js", "nextjs", "nuxt",
        "django", "flask", "fastapi", "spring", "spring boot", "express",
        "nestjs", "laravel", "rails", "asp.net", "fastify"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "postgres", "sqlite", "mongodb",
        "redis", "elasticsearch", "cassandra", "dynamodb", "neo4j",
        "oracle", "mssql", "couchdb", "firebase", "supabase"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
        "circleci", "helm", "prometheus", "grafana", "nginx", "linux",
        "ci/cd", "devops", "datadog"
    ],
    "Machine Learning & AI": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas",
        "numpy", "opencv", "nlp", "bert", "gpt", "llm", "deep learning",
        "machine learning", "computer vision", "transformers", "hugging face",
        "xgboost", "lightgbm", "mlflow", "airflow", "spark", "hadoop"
    ],
    "Tools & Methodologies": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence",
        "agile", "scrum", "kanban", "tdd", "rest", "graphql", "grpc",
        "microservices", "api", "swagger", "openapi", "postman"
    ],
    "Soft Skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "project management", "collaboration"
    ]
}

# Build a reverse lookup: skill → category
# e.g., "python" → "Programming Languages"
# This lets us look up any skill's category in O(1) time
SKILL_CATEGORY_MAP: Dict[str, str] = {
    skill: category
    for category, skills in SKILLS_DB.items()
    for skill in skills
}


def extract_skills(text: str) -> Set[str]:
    """
    Scan text for known skill keywords.

    Uses regex with word boundaries (\\b) to avoid false matches:
      - "go" won't match inside "algorithm" or "Google"
      - "r" won't match inside "requirements"

    Args:
        text: Resume or JD text (any case)

    Returns:
        Set of matched skill names (lowercase)
    """
    found: Set[str] = set()
    text_lower = text.lower()

    for skills in SKILLS_DB.values():
        for skill in skills:
            # Use word boundaries for short/ambiguous skills
            # For longer phrases (e.g., "spring boot"), simple substring is fine
            if len(skill) <= 3 or " " in skill:
                pattern = r'\b' + re.escape(skill) + r'\b'
            else:
                pattern = re.escape(skill)

            if re.search(pattern, text_lower):
                found.add(skill)

    logger.debug(f"Skills found: {len(found)}")
    return found


def _count_occurrences(skill: str, text: str) -> int:
    """
    Count how many times a skill appears in the text.
    Used for priority determination: more mentions = higher priority.

    Args:
        skill: Skill name to search for (case-insensitive)
        text:  Text to search within

    Returns:
        Integer count of occurrences
    """
    return len(re.findall(re.escape(skill.lower()), text.lower()))


def determine_priority(skill: str, jd_text: str) -> Priority:
    """
    Assign a priority level to a missing skill.

    ALGORITHM:
      1. Check if the skill appears near "required/must/essential" → HIGH
      2. Check if the skill appears near "preferred/nice to have" → LOW
      3. Fall back to frequency: 3+ mentions → HIGH, 1-2 → MEDIUM

    Args:
        skill:   The missing skill name
        jd_text: Full job description text

    Returns:
        Priority.HIGH, Priority.MEDIUM, or Priority.LOW
    """
    jd_lower = jd_text.lower()
    skill_lower = skill.lower()

    # Find where the skill appears and look at surrounding text (±120 chars)
    pos = jd_lower.find(skill_lower)
    if pos != -1:
        ctx_start = max(0, pos - 120)
        ctx_end   = min(len(jd_lower), pos + len(skill) + 120)
        context   = jd_lower[ctx_start:ctx_end]

        # HIGH priority signals
        HIGH_SIGNALS = [
            "required", "must have", "must-have", "essential",
            "mandatory", "critical", "strong", "expertise in",
            "proficient in"
        ]

        # LOW priority signals (check first so 'Preferred: experience with' is LOW)
        LOW_SIGNALS = [
            "preferred", "nice to have", "nice-to-have", "bonus",
            "plus", "advantage", "optional", "desirable", "a plus"
        ]
        for signal in LOW_SIGNALS:
            if signal in context:
                return Priority.LOW

        for signal in HIGH_SIGNALS:
            if signal in context:
                return Priority.HIGH

    # Fallback: frequency-based priority
    freq = _count_occurrences(skill, jd_text)
    if freq >= 3:
        return Priority.HIGH
    elif freq >= 1:
        return Priority.MEDIUM
    else:
        return Priority.LOW   # rare fallback


def generate_action_item(skill: str, category: str) -> str:
    """
    Generate a specific, actionable improvement suggestion.

    Each category gets a tailored template because the right action
    for learning SQL is different from the right action for learning
    Kubernetes or improving communication skills.

    Args:
        skill:    The missing skill name
        category: The skill's category from SKILLS_DB

    Returns:
        A concrete action item string
    """
    # Capitalize skill for clean display (e.g., "python" → "Python")
    s = skill.upper() if len(skill) <= 3 else skill.title()

    templates: Dict[str, str] = {
        "Programming Languages": (
            f"Learn {s} via official docs or freeCodeCamp; "
            f"add a small project to your GitHub; "
            f"list in 'Technical Skills' section"
        ),
        "Web Frameworks": (
            f"Build a CRUD app using {s} and host it on GitHub Pages or Vercel; "
            f"mention in 'Projects' section"
        ),
        "Databases": (
            f"Practice {s} on SQLZoo or LeetCode (Database problems); "
            f"add to 'Database Skills' in your resume"
        ),
        "Cloud & DevOps": (
            f"Create a free-tier account and complete the official {s} getting-started; "
            f"consider an official certification if HIGH priority"
        ),
        "Machine Learning & AI": (
            f"Complete a {s} project on Kaggle or Google Colab; "
            f"document results and add to 'ML Projects' section"
        ),
        "Tools & Methodologies": (
            f"Integrate {s} into your next project workflow; "
            f"add to 'Tools & Technologies' section of resume"
        ),
        "Soft Skills": (
            f"Highlight {s} in work experience bullets with a specific example; "
            f"e.g., 'Led cross-functional team of 5 to deliver X'"
        ),
    }

    return templates.get(
        category,
        f"Add '{s}' to your resume's relevant skills section; "
        f"complete a small practice project if unfamiliar"
    )


def filter_or_groups(missing_skills: Set[str], jd_text: str, resume_skills: Set[str]) -> Set[str]:
    """
    Handle "one or more" / "or" conditions in the Job Description.
    If the JD asks for "Java, Python, OR C++" and the resume has Python,
    we shouldn't flag Java and C++ as gaps.
    """
    # Split text into sentences or bullet points
    sentences = re.split(r'[.\n;•]', jd_text.lower())
    
    satisfied_or_skills: Set[str] = set()
    strict_skills: Set[str] = set()
    
    # Keywords that indicate an "OR" group
    or_keywords = ["one or more", "at least one", "either", " or "]
    
    # Keywords that indicate a strict requirement
    strict_keywords = ["must", "required", "essential", "mandatory"]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_skills = extract_skills(sentence)
        if not sentence_skills:
            continue
            
        is_or_group = any(kw in sentence for kw in or_keywords)
        is_strict = any(kw in sentence for kw in strict_keywords)
        
        if is_or_group:
            # If the resume has at least one skill from this OR group...
            if sentence_skills & resume_skills:
                # The group is satisfied! Mark these skills to be potentially removed from gaps
                satisfied_or_skills.update(sentence_skills)
        elif is_strict and not is_or_group:
            # These skills are strictly required in this sentence (and not an OR condition)
            strict_skills.update(sentence_skills)
            
    # We only remove satisfied OR skills from missing_skills if they aren't STRICTLY required elsewhere
    skills_to_remove = satisfied_or_skills - strict_skills

    # Also process explicit "including" / "such as" lists which may span
    # comma-separated values and use slashes (e.g., "C/C++"). Treat each
    # discovered list as an OR group: if resume matches any, remove the rest.
    try:
        explicit_groups = _extract_including_groups(jd_text)
        for grp in explicit_groups:
            grp_set = set(grp)
            if grp_set & resume_skills:
                # Only mark group skills as satisfied if not marked strict elsewhere
                satisfied_or_skills.update(grp_set)
    except Exception:
        # Be defensive — if helper fails for any reason, don't alter results
        logger.debug("Failed to parse explicit including-groups in JD text")

    skills_to_remove = satisfied_or_skills - strict_skills

    return missing_skills - skills_to_remove


def _extract_including_groups(jd_text: str) -> Set[frozenset]:
        """
        Find explicit "including" or "such as" lists and return groups of skills.

        Example matches:
            "including: Java, C/C++, C#, Python, JavaScript, Go, or Rust"
            "such as Python, JavaScript or Go"

        Returns a set of frozensets where each frozenset contains the skills found
        in that list (lowercased, as in SKILLS_DB).
        """
        groups = set()
        lower = jd_text.lower()

        include_patterns = [r'including but not limited to', r'including', r'such as', r'for example']
        for pat in include_patterns:
                for m in re.finditer(pat, lower):
                        # Take text after the match until sentence end
                        start = m.end()
                        end_match = re.search(r'[.\n;•]', lower[start:])
                        end = start + (end_match.start() if end_match else len(lower[start:]))
                        fragment = lower[start:end]

                        # Normalize slashes and ' or ' separators into commas for easier splitting
                        fragment = fragment.replace('/', ',')
                        fragment = fragment.replace(' or ', ',')
                        fragment = fragment.replace(';', ',')

                        # Extract skills from the fragment using existing extractor
                        skills_found = extract_skills(fragment)
                        if skills_found:
                                groups.add(frozenset(skills_found))

        return groups


def run_gap_analysis(resume_text: str, jd_text: str) -> dict:
    """
    Main gap analysis entry point.

    Full pipeline:
      1. Extract skills from resume
      2. Extract skills from JD
      3. Compute intersection (matches) and difference (gaps)
      4. Assign priorities to gaps
      5. Generate action items
      6. Sort by priority

    Args:
        resume_text: Cleaned resume text
        jd_text:     Cleaned job description text

    Returns:
        dict with keys:
          - skill_gaps       → List[SkillGap]
          - resume_skills    → List[str] (sorted, title case)
          - jd_skills        → List[str] (sorted, title case)
          - matching_skills  → List[str] (sorted, title case)
    """
    logger.info("Running gap analysis...")

    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(jd_text)

    matching_skills = resume_skills & jd_skills        # Set intersection
    missing_skills  = jd_skills - resume_skills         # Set difference

    # Filter out skills that are part of satisfied "OR" groups
    missing_skills = filter_or_groups(missing_skills, jd_text, resume_skills)

    logger.info(
        f"Resume: {len(resume_skills)} skills | "
        f"JD: {len(jd_skills)} skills | "
        f"Match: {len(matching_skills)} | "
        f"Gaps: {len(missing_skills)}"
    )

    # Build SkillGap objects for each missing skill
    gaps: List[SkillGap] = []
    for skill in missing_skills:
        category = SKILL_CATEGORY_MAP.get(skill, "General")
        priority = determine_priority(skill, jd_text)
        action   = generate_action_item(skill, category)
        skill_encoded = urllib.parse.quote(skill)

        gaps.append(SkillGap(
            skill=skill.title(),
            priority=priority,
            action_item=action,
            category=category,
            free_resource=f"https://www.youtube.com/results?search_query={skill_encoded}+full+course",
            paid_resource=f"https://www.coursera.org/search?query={skill_encoded}"
        ))

    # Sort: HIGH → MEDIUM → LOW, then alphabetically within each tier
    priority_rank = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    gaps.sort(key=lambda g: (priority_rank[g.priority], g.skill))

    return {
        "skill_gaps":       gaps,
        "resume_skills":    sorted(s.title() for s in resume_skills),
        "jd_skills":        sorted(s.title() for s in jd_skills),
        "matching_skills":  sorted(s.title() for s in matching_skills),
    }
