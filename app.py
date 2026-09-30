import json
import os
import re
import time
from typing import Any

import streamlit as st
from docx import Document
from google import genai
from pypdf import PdfReader


# ============================================================
# CONFIGURATION
# ============================================================

PRIMARY_MODEL = "gemini-3.5-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

MAX_FILE_MB = 10
MAX_RETRIES = 3


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="AI Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# GEMINI API KEY
# ============================================================

def get_api_key() -> str | None:

    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    return key or os.getenv("GEMINI_API_KEY")


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file) -> str:

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        text = page.extract_text() or ""

        if text.strip():
            pages.append(text)

    return "\n".join(pages).strip()


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx_text(uploaded_file) -> str:

    document = Document(uploaded_file)

    parts = []

    # Paragraphs
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
# RESUME EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file) -> str:

    extension = uploaded_file.name.lower().split(".")[-1]

    if extension == "pdf":

        return extract_pdf_text(uploaded_file)

    if extension == "docx":

        return extract_docx_text(uploaded_file)

    raise ValueError(
        "Only PDF and DOCX files are supported."
    )


# ============================================================
# CLEAN JSON
# ============================================================

def clean_json_text(text: str) -> str:

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
# NORMALIZE RESULT
# ============================================================

def normalize_result(
    data: dict[str, Any]
) -> dict[str, Any]:

    categories = data.get(
        "categories",
        {}
    ) or {}

    expected_categories = [
        "keyword_match",
        "formatting_ats",
        "sections",
        "experience_impact",
        "skills",
        "contact_information",
        "readability",
    ]

    normalized_categories = {}

    for category in expected_categories:

        item = categories.get(
            category,
            {}
        ) or {}

        try:

            score = int(
                float(
                    item.get(
                        "score",
                        0
                    )
                )
            )

        except (TypeError, ValueError):

            score = 0

        score = max(
            0,
            min(
                100,
                score
            )
        )

        normalized_categories[category] = {

            "score": score,

            "feedback": str(
                item.get(
                    "feedback",
                    "No feedback provided."
                )
            ),
        }

    try:

        overall_score = int(
            float(
                data.get(
                    "overall_score",
                    0
                )
            )
        )

    except (TypeError, ValueError):

        overall_score = round(
            sum(
                item["score"]
                for item in normalized_categories.values()
            )
            /
            len(normalized_categories)
        )

    overall_score = max(
        0,
        min(
            100,
            overall_score
        )
    )

    return {

        "overall_score":
            overall_score,

        "categories":
            normalized_categories,

        "strengths": [
            str(x)
            for x in (
                data.get(
                    "strengths",
                    []
                )
                or []
            )
        ][:8],

        "improvements": [
            str(x)
            for x in (
                data.get(
                    "improvements",
                    []
                )
                or []
            )
        ][:10],

        "missing_keywords": [
            str(x)
            for x in (
                data.get(
                    "missing_keywords",
                    []
                )
                or []
            )
        ][:20],

        "ats_warnings": [
            str(x)
            for x in (
                data.get(
                    "ats_warnings",
                    []
                )
                or []
            )
        ][:10],

        "summary":
            str(
                data.get(
                    "summary",
                    ""
                )
            ),
    }


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    resume_text: str,
    job_description: str
) -> str:

    if job_description.strip():

        job_context = job_description.strip()

    else:

        job_context = (
            "No job description was provided. "
            "Evaluate general ATS readiness."
        )

    return f"""
You are an expert resume ATS analyzer.

Analyze the resume and provide an AI-based ATS
readiness assessment.

IMPORTANT:
This is an estimated ATS readiness score.
It is NOT an exact score from a specific employer's ATS.

Give an overall score from 0 to 100.

Evaluate these categories:

1. keyword_match
2. formatting_ats
3. sections
4. experience_impact
5. skills
6. contact_information
7. readability

If a job description is provided,
keyword_match should compare the resume
against that job description.

If no job description is provided,
evaluate general use of relevant professional terminology.

Do NOT invent:

- experience
- education
- employers
- dates
- certifications
- skills
- achievements

Only use information actually present in the resume.

Identify:

1. Resume strengths.
2. Specific improvements.
3. Missing or weak keywords.
4. ATS warnings.
5. Overall summary.

Possible ATS warnings include:

- tables
- columns
- graphics
- images
- unusual symbols
- unclear headings
- unclear dates
- headers/footers
- excessive formatting

IMPORTANT:
Text extraction cannot always detect visual formatting.
Do not claim something is present unless there is evidence.

Return ONLY valid JSON.

Use exactly this structure:

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

RESUME:

{resume_text}
"""


# ============================================================
# GEMINI REQUEST WITH RETRIES
# ============================================================

def call_gemini_with_retry(
    client,
    model_name: str,
    prompt: str
):

    last_error = None

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = client.models.generate_content(

                model=model_name,

                contents=prompt,

                config={
                    "response_mime_type":
                        "application/json"
                },
            )

            return response

        except Exception as exc:

            last_error = exc

            error_text = str(exc).lower()

            is_temporary = (
                "503" in error_text
                or
                "unavailable" in error_text
                or
                "high demand" in error_text
                or
                "429" in error_text
                or
                "resource exhausted" in error_text
            )

            if not is_temporary:

                raise

            if attempt < MAX_RETRIES - 1:

                wait_seconds = 2 ** attempt

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        "Gemini API is temporarily unavailable "
        "after multiple retry attempts."
        f"\n\nLast error: {last_error}"
    )


# ============================================================
# ANALYZE RESUME
# ============================================================

def analyze_resume(
    resume_text: str,
    job_description: str,
    api_key: str
):

    client = genai.Client(
        api_key=api_key
    )

    prompt = build_prompt(
        resume_text,
        job_description
    )

    # --------------------------------------------------------
    # TRY PRIMARY MODEL
    # --------------------------------------------------------

    try:

        response = call_gemini_with_retry(
            client,
            PRIMARY_MODEL,
            prompt
        )

        model_used = PRIMARY_MODEL

    except Exception as primary_error:

        # ----------------------------------------------------
        # FALLBACK MODEL
        # ----------------------------------------------------

        try:

            response = call_gemini_with_retry(
                client,
                FALLBACK_MODEL,
                prompt
            )

            model_used = FALLBACK_MODEL

        except Exception as fallback_error:

            raise RuntimeError(
                "Both Gemini models were temporarily "
                "unavailable.\n\n"
                f"Primary model error:\n"
                f"{primary_error}\n\n"
                f"Fallback model error:\n"
                f"{fallback_error}"
            )

    raw_response = clean_json_text(
        response.text or ""
    )

    try:

        result = json.loads(
            raw_response
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Gemini returned an invalid JSON response."
        ) from exc

    normalized = normalize_result(
        result
    )

    normalized["_model_used"] = model_used

    return normalized


# ============================================================
# SCORE LABEL
# ============================================================

def score_label(
    score: int
) -> str:

    if score >= 85:

        return "Strong ATS readiness"

    if score >= 70:

        return "Good, but needs improvement"

    if score >= 50:

        return "Needs improvement"

    return "Major improvements recommended"


# ============================================================
# TITLE
# ============================================================

st.title(
    "📄 AI Resume ATS Analyzer"
)

st.write(
    "Upload your resume and receive an "
    "AI-assisted ATS readiness score, "
    "keyword analysis, warnings, and "
    "specific improvement suggestions."
)

st.info(
    "The score is an AI-based estimate. "
    "It is not a guarantee of how a particular "
    "employer's ATS will score your resume."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Settings"
    )

    st.caption(
        f"Primary model: `{PRIMARY_MODEL}`"
    )

    st.caption(
        f"Fallback model: `{FALLBACK_MODEL}`"
    )

    st.markdown(
        "**Supported files:** PDF and DOCX"
    )

    st.markdown(
        f"**Maximum file size:** "
        f"{MAX_FILE_MB} MB"
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(

    "📎 Upload your resume",

    type=[
        "pdf",
        "docx"
    ]
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
        "The ATS keyword analysis will be "
        "tailored to this specific job."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_clicked = st.button(

    "🔍 Analyze Resume",

    type="primary",

    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_clicked:

    if uploaded_file is None:

        st.error(
            "Please upload a PDF or DOCX resume."
        )

        st.stop()


    if uploaded_file.size > (
        MAX_FILE_MB * 1024 * 1024
    ):

        st.error(
            f"File is larger than "
            f"{MAX_FILE_MB} MB."
        )

        st.stop()


    api_key = get_api_key()

    if not api_key:

        st.error(
            "Gemini API key is missing."
        )

        st.stop()


    with st.spinner(
        "📊 Reading resume and generating ATS analysis..."
    ):

        try:

            resume_text = extract_resume_text(
                uploaded_file
            )

            if len(
                resume_text.strip()
            ) < 80:

                st.error(
                    "Very little text could be "
                    "extracted from this resume."
                )

                st.warning(
                    "If your PDF is scanned/image-only, "
                    "try uploading a text-based PDF "
                    "or DOCX file."
                )

                st.stop()


            result = analyze_resume(

                resume_text,

                job_description,

                api_key
            )


            st.session_state[
                "ats_result"
            ] = result

            st.session_state[
                "resume_name"
            ] = uploaded_file.name


        except Exception as exc:

            st.error(
                "Analysis failed."
            )

            st.exception(exc)

            st.info(
                "If the error is 503 UNAVAILABLE or "
                "high demand, Gemini's service is "
                "temporarily unavailable. "
                "The application now retries automatically "
                "and uses a fallback Flash-Lite model."
            )

            st.stop()


# ============================================================
# RESULTS
# ============================================================

if "ats_result" in st.session_state:

    result = st.session_state[
        "ats_result"
    ]

    st.divider()

    st.subheader(
        f"📊 Results — "
        f"{st.session_state.get('resume_name', 'Resume')}"
    )


    # --------------------------------------------------------
    # MODEL USED
    # --------------------------------------------------------

    model_used = result.get(
        "_model_used"
    )

    if model_used:

        st.caption(
            f"Analysis model: `{model_used}`"
        )


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = result[
        "overall_score"
    ]

    col1, col2 = st.columns(
        [1, 2]
    )

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


    # --------------------------------------------------------
    # CATEGORY SCORES
    # --------------------------------------------------------

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


    columns = st.columns(
        4
    )


    for index, (
        key,
        label
    ) in enumerate(
        category_names.items()
    ):

        item = result[
            "categories"
        ][key]

        with columns[
            index % 4
        ]:

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


    # --------------------------------------------------------
    # TWO COLUMNS
    # --------------------------------------------------------

    left, right = st.columns(
        2
    )


    # --------------------------------------------------------
    # STRENGTHS
    # --------------------------------------------------------

    with left:

        st.subheader(
            "✅ Resume Strengths"
        )

        if result[
            "strengths"
        ]:

            for item in result[
                "strengths"
            ]:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "No strengths returned."
            )


        # ----------------------------------------------------
        # IMPROVEMENTS
        # ----------------------------------------------------

        st.subheader(
            "🔧 Recommended Improvements"
        )

        if result[
            "improvements"
        ]:

            for item in result[
                "improvements"
            ]:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "No improvements returned."
            )


    # --------------------------------------------------------
    # KEYWORDS / WARNINGS
    # --------------------------------------------------------

    with right:

        st.subheader(
            "🔑 Missing / Weak Keywords"
        )

        if result[
            "missing_keywords"
        ]:

            for item in result[
                "missing_keywords"
            ]:

                st.markdown(
                    f"- `{item}`"
                )

        else:

            st.write(
                "No important missing keywords identified."
            )


        st.subheader(
            "⚠️ ATS Warnings"
        )

        if result[
            "ats_warnings"
        ]:

            for item in result[
                "ats_warnings"
            ]:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "No major ATS warnings identified."
            )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "📝 Overall Summary"
    )

    st.write(
        result[
            "summary"
        ]
        or
        "No summary returned."
    )


    # --------------------------------------------------------
    # PRIVACY
    # --------------------------------------------------------

    st.caption(
        "Privacy note: resume content is sent to "
        "the Gemini API for analysis. Do not upload "
        "documents you are not authorized to process."
    )
