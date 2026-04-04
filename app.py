import os
import json
import anthropic
import pypdf
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1) API key from Streamlit Secrets
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# 2) Embeddings for RAG
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_store(texts):
    vectors = embedding_model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, texts

def retrieve_context(query, texts, index):
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, k=2)
    return "\n".join([texts[i] for i in indices[0]])

# 3) Extract text from uploaded PDF
def extract_resume_text(uploaded_file):
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# 4) Resume heading check
def check_resume_headings(resume_text):
    standard_headings = {
        "work experience": ["work experience", "experience", "professional experience", "employment history"],
        "skills":          ["skills", "technical skills", "core competencies"],
        "education":       ["education", "academic background", "qualifications"],
        "summary":         ["summary", "objective", "professional summary", "about me", "profile"],
        "certifications":  ["certifications", "certificates", "licenses"]
    }
    found = [h for h, variants in standard_headings.items()
             if any(v in resume_text.lower() for v in variants)]
    return found, len(found) < 3

# 5) Analyze resume using Claude
def analyze_resume(job_desc, resume_text):
    index, texts = build_vector_store([job_desc, resume_text])
    context = retrieve_context(job_desc, texts, index)

    prompt = f"""
You are an honest career coach and ATS expert. Review this resume against the job description and provide fair, specific feedback.

Provide the following:

1. ATS match score (0-100) based on hard skills, experience relevance, and industry alignment.
   - Penalize heavily for missing critical skills, vague bullet points, or irrelevant experience.
   - Only give 80+ if the resume genuinely competes for this role.

2. Honest, balanced overall feedback — acknowledge what the resume does well for this role, but be clear and specific about what is missing or weak. Do not exaggerate strengths. If a critical skill or keyword is absent, call it out directly without softening it.

3. Up to 5 missing high-priority keywords from the job description not found in the resume, with a specific suggestion on exactly where and how to add each one naturally.

Context:
{context}

Output ONLY a valid JSON object with no markdown fences or extra text:
{{
  "match_score": <int>,
  "overall_feedback": "<string — honest and balanced, 3-5 sentences>",
  "missing_keywords": [{{"keyword": "<string>", "suggestion": "<string>"}}]
}}
"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()

    try:
        clean = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON output", "raw_response": response_text}


# 6) Streamlit UI
st.set_page_config(page_title="ATS Resume Optimizer", page_icon="📄")
st.title("📄 ATS Resume Optimizer")
st.write("Upload your resume and paste a job description to get an honest ATS score and feedback.")

job_description = st.text_area("Paste Job Description", height=250, placeholder="Paste the full job description here...")
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Analyze Resume"):
    if not job_description.strip():
        st.warning("Please paste a job description.")
    elif uploaded_file is None:
        st.warning("Please upload a resume PDF.")
    else:
        with st.spinner("Analyzing with Claude..."):
            resume_text = extract_resume_text(uploaded_file)
            headings, headings_warning = check_resume_headings(resume_text)
            result = analyze_resume(job_description, resume_text)

        # Heading warning
        if headings_warning:
            st.warning(f"⚠️ Only {len(headings)} standard headings found: {headings}. Consider adding: Work Experience, Skills, Education, Summary, or Certifications.")

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # Match score
            score = result["match_score"]
            st.subheader("ATS Match Score")
            color = "green" if score >= 75 else "orange" if score >= 50 else "red"
            st.markdown(f"<h1 style='color:{color}'>{score} / 100</h1>", unsafe_allow_html=True)
            st.progress(score / 100)

            # Balanced feedback
            st.subheader("💬 Feedback")
            st.warning(result.get("overall_feedback", "No feedback returned."))

            # Missing keywords
            st.subheader("Missing Keywords")
            for item in result.get("missing_keywords", []):
                with st.expander(f"🔑 {item['keyword']}"):
                    st.write(item["suggestion"])
