import json
import os
import re
from typing import Any

import streamlit as st
from docx import Document
from google import genai
from pypdf import PdfReader


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash"
MAX_FILE_MB = 10


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str | None:
    """
    Read Gemini API key from Streamlit Secrets first,
    then from environment variables.
    """

    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    return key or os.getenv("GEMINI_API_KEY")


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file) -> str:
    """
    Extract text from PDF resume.
    """

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()


# ============================================================
# DOCX TEXT EXTRACTION
# ============================================================

def extract_docx_text(uploaded_file) -> str:
    """
    Extract text from DOCX resume.
    """

    document = Document(uploaded_file)

    parts = []

    # Normal paragraphs
    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            parts.append(text)

    # Tables
    for table in document.tables:

        for row in table.rows:

            row_text = " | ".join(
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            )

            if row_text:
                parts.append(row_text)

    return "\n".join(parts).strip()


# ============================================================
# RESUME TEXT EXTRACTION ROUTER
# ============================================================

def extract_resume_text(uploaded_file) -> str:

    suffix = uploaded_file.name.lower().split(".")[-1]

    if suffix == "pdf":
        return extract_pdf_text(uploaded_file)

    if suffix == "docx":
        return extract_docx_text(uploaded_file)

    raise ValueError(
        "Unsupported file type. Please upload a PDF or DOCX file."
    )


# ============================================================
# CLEAN GEMINI JSON RESPONSE
# ============================================================

def clean_json_text(text: str) -> str:
    """
    Remove Markdown code fences if Gemini returns them.
    """

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


# ============================================================
# NORMALIZE AI RESULT
# ============================================================

def normalize_result(data: dict[str, Any]) -> dict[str, Any]:

    categories = data.get("categories", {}) or {}

    expected = [
        "keyword_match",
        "formatting_ats",
        "sections",
        "experience_impact",
        "skills",
        "contact_information",
        "readability",
    ]

    normalized_categories = {}

    for name in expected:

        item = categories.get(name, {}) or {}

        try:

            score = max(
                0,
                min(
                    100,
                    int(float(item.get("score", 0)))
                )
            )

        except (TypeError, ValueError):

            score = 0

        normalized_categories[name] = {
            "score": score,
            "feedback": str(
                item.get(
                    "feedback",
                    "No feedback provided."
                )
            ),
        }

    try:

        overall = max(
            0,
            min(
                100,
                int(float(data.get("overall_score", 0)))
            )
        )

    except (TypeError, ValueError):

        overall = round(
            sum(
                x["score"]
                for x in normalized_categories.values()
            )
            / len(normalized_categories)
        )

    return {
        "overall_score": overall,

        "categories": normalized_categories,

        "strengths": [
            str(x)
            for x in (data.get("strengths", []) or [])
        ][:8],

        "improvements": [
            str(x)
            for x in (data.get("improvements", []) or [])
        ][:10],

        "missing_keywords": [
            str(x)
            for x in (data.get("missing_keywords", []) or [])
        ][:20],

        "ats_warnings": [
            str(x)
            for x in (data.get("ats_warnings", []) or [])
        ][:10],

        "summary": str(
            data.get("summary", "")
        ),
    }


# ============================================================
# GEMINI RESUME ANALYSIS
# ============================================================

def analyze_resume(
    resume_text: str,
    job_description: str,
    api_key: str,
) -> dict[str, Any]:

    client = genai.Client(
        api_key=api_key
    )

    if job_description.strip():

        job_context = job_description.strip()

    else:

        job_context = (
            "No job description was provided. "
            "Evaluate general ATS readiness."
        )

    prompt = f"""
You are an expert resume ATS analyzer and career-document reviewer.

Analyze the resume below and provide a practical ATS-readiness assessment.

IMPORTANT:
This is an AI-based ATS estimate. It is NOT a guarantee of how
a particular company's ATS will score the resume.

SCORING:

Give an overall score from 0 to 100.

Score these seven categories from 0 to 100:

1. keyword_match
2. formatting_ats
3. sections
4. experience_impact
5. skills
6. contact_information
7. readability

If a job description is supplied:

keyword_match should measure how well the resume matches
the job description.

If no job description is supplied:

keyword_match should measure general clarity and
relevant terminology.

DO NOT invent:

- work experience
- qualifications
- employers
- dates
- achievements
- certifications
- skills

as if they already exist in the resume.

REVIEW THE FOLLOWING:

1. Identify strengths actually supported by the resume.

2. Give specific improvements.

Explain what should be changed.

Where useful, provide short example wording,
but do not invent facts.

3. Identify important keywords from the job description
that appear to be missing or weakly represented.

4. Identify ATS warnings such as:

- unusual formatting
- tables
- columns
- graphics
- images
- unusual symbols
- missing standard headings
- unclear dates
- headers/footers
- excessive formatting

IMPORTANT:

Text extraction cannot reliably detect every visual
formatting problem.

Only claim a formatting problem when there is
reasonable evidence.

5. Give a short overall summary.

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
  "overall_score": 0,

  "categories": {{

    "keyword_match": {{
      "score": 0,
      "feedback": ""
    }},

    "formatting_ats": {{
      "score": 0,
      "feedback": ""
    }},

    "sections": {{
      "score": 0,
      "feedback": ""
    }},

    "experience_impact": {{
      "score": 0,
      "feedback": ""
    }},

    "skills": {{
      "score": 0,
      "feedback": ""
    }},

    "contact_information": {{
      "score": 0,
      "feedback": ""
    }},

    "readability": {{
      "score": 0,
      "feedback": ""
    }}

  }},

  "strengths": [],

  "improvements": [],

  "missing_keywords": [],

  "ats_warnings": [],

  "summary": ""
}}

JOB DESCRIPTION:

{job_context}

RESUME TEXT:

{resume_text}
"""

    response = client.models.generate_content(

        model=MODEL_NAME,

        contents=prompt,

        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    raw = clean_json_text(
        response.text or ""
    )

    try:

        parsed = json.loads(raw)

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Gemini returned invalid JSON. "
            f"Response: {raw[:1000]}"
        ) from exc

    return normalize_result(parsed)


# ============================================================
# SCORE LABEL
# ============================================================

def score_label(score: int) -> str:

    if score >= 85:
        return "Strong ATS readiness"

    if score >= 70:
        return "Good, but needs improvement"

    if score >= 50:
        return "Needs improvement"

    return "Major improvements recommended"


# ============================================================
# APPLICATION UI
# ============================================================

st.title("📄 AI Resume ATS Analyzer")

st.write(
    "Upload your resume to receive an AI-assisted "
    "ATS readiness score and specific improvement suggestions."
)

st.info(
    "The score is an AI-based estimate, not a guarantee "
    "of how a specific employer's ATS will score your resume."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.caption(
        f"Gemini model: `{MODEL_NAME}`"
    )

    st.markdown(
        "**Supported files:** PDF and DOCX"
    )

    st.markdown(
        f"**Maximum upload size:** {MAX_FILE_MB} MB"
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📎 Upload your resume",
    type=["pdf", "docx"],
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

job_description = st.text_area(

    "💼 Job Description "
    "(Optional but recommended)",

    height=220,

    placeholder=(
        "Paste the job description here. "
        "The ATS keyword analysis will then be "
        "tailored to this specific job."
    ),
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_clicked = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_clicked:

    # Check file
    if uploaded_file is None:

        st.error(
            "Please upload a PDF or DOCX resume first."
        )

        st.stop()

    # Check size
    if uploaded_file.size > MAX_FILE_MB * 1024 * 1024:

        st.error(
            f"File is too large. "
            f"Please upload a file smaller than "
            f"{MAX_FILE_MB} MB."
        )

        st.stop()

    # Check API key
    api_key = get_api_key()

    if not api_key:

        st.error(
            "Gemini API key is missing. "
            "Add GEMINI_API_KEY in Streamlit Secrets "
            "or as an environment variable."
        )

        st.stop()

    with st.spinner(
        "📊 Reading resume and generating ATS analysis..."
    ):

        try:

            # Extract text
            resume_text = extract_resume_text(
                uploaded_file
            )

            # Check extracted text
            if len(resume_text.strip()) < 80:

                st.error(
                    "Very little text could be extracted "
                    "from this file."
                )

                st.warning(
                    "If this is a scanned/image-only PDF, "
                    "use a text-based PDF or DOCX version."
                )

                st.stop()

            # Gemini analysis
            result = analyze_resume(
                resume_text,
                job_description,
                api_key,
            )

            # Save results
            st.session_state["ats_result"] = result

            st.session_state["resume_name"] = (
                uploaded_file.name
            )

        except Exception as exc:

            st.error(
                f"Analysis failed: {exc}"
            )

            st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "ats_result" in st.session_state:

    result = st.session_state["ats_result"]

    st.divider()

    st.subheader(
        f"📊 Results — "
        f"{st.session_state.get('resume_name', 'Resume')}"
    )


    # ========================================================
    # OVERALL SCORE
    # ========================================================

    score = result["overall_score"]

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric(
            "ATS Score",
            f"{score}/100"
        )

    with col2:

        st.progress(
            score / 100
        )

        st.write(
            f"**{score_label(score)}**"
        )


    # ========================================================
    # CATEGORY SCORES
    # ========================================================

    st.subheader(
        "📈 Category Scores"
    )

    category_names = {

        "keyword_match":
            "Keyword Match",

        "formatting_ats":
            "ATS Formatting",

        "sections":
            "Resume Sections",

        "experience_impact":
            "Experience Impact",

        "skills":
            "Skills",

        "contact_information":
            "Contact Information",

        "readability":
            "Readability",
    }


    cols = st.columns(4)

    for index, (key, label) in enumerate(
        category_names.items()
    ):

        item = result["categories"][key]

        with cols[index % 4]:

            st.metric(
                label,
                f"{item['score']}/100"
            )

            st.progress(
                item["score"] / 100
            )

            st.caption(
                item["feedback"]
            )


    # ========================================================
    # TWO COLUMN RESULTS
    # ========================================================

    left, right = st.columns(2)


    # ========================================================
    # STRENGTHS + IMPROVEMENTS
    # ========================================================

    with left:

        st.subheader(
            "✅ Resume Strengths"
        )

        if result["strengths"]:

            for item in result["strengths"]:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "No strengths were returned."
            )


        st.subheader(
            "🔧 Recommended Improvements"
        )

        if result["improvements"]:

            for item in result["improvements"]:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "No improvements were returned."
            )


    # ========================================================
    # KEYWORDS + WARNINGS
    # ========================================================

    with right:

        st.subheader(
            "🔑 Missing / Weak Keywords"
        )

        if result["missing_keywords"]:

            for item in result["missing_keywords"]:

                st.markdown(
                    f"- `{item}`"
                )

        else:

            st.write(
                "No important missing keywords were identified."
            )


        st.subheader(
            "⚠️ ATS Warnings"
        )

        if result["ats_warnings"]:

            for item in result["ats_warnings"]:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "No major ATS warnings were identified "
                "from the available text."
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader(
        "📝 Overall Summary"
    )

    st.write(
        result["summary"]
        or
        "No summary was returned."
    )


    # ========================================================
    # PRIVACY MESSAGE
    # ========================================================

    st.caption(
        "Privacy note: this app sends the uploaded "
        "resume content to the Gemini API for analysis. "
        "Do not upload documents you are not authorized "
        "to process."
    )
