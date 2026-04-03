# required packages — run these in a separate cell first:
# !pip install anthropic pypdf sentence-transformers faiss-cpu

# 1) Importing required libraries
import os
import json
import anthropic
import pypdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 2) Configuring Claude API via Kaggle Secrets
# In Kaggle: Add-ons → Secrets → Add → Name: ANTHROPIC_API_KEY, Value: your key
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["ANTHROPIC_API_KEY"] = user_secrets.get_secret("ANTHROPIC_API_KEY")
except:
    # Fallback for local use — set env variable externally, never hardcode
    pass

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# 3) Embeddings for RAG (no langchain needed)
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

# 4) Paste your job description here
job_description = """
Responsibilities:

Assist in creating and maintaining interactive dashboards and visual reports using Power BI.
Analyze data sets to identify trends, patterns, and actionable insights.
Develop SQL queries to extract and manipulate data from relational databases.
Collaborate with team members to gather requirements and understand reporting needs.
Support the analysis and interpretation of data to inform business decisions.
Perform data cleaning and validation to ensure data accuracy.
Document data sources, methods, and visualization processes.
Participate in team meetings to present findings and recommendations.

Qualifications:

Currently pursuing a degree in Data Science, Business Analytics, Computer Science, or a related field.
Experience with Power BI for data visualization and reporting.
Basic knowledge of SQL for querying and data manipulation.
Understanding of relational databases and data modeling concepts.
Strong analytical thinking and problem-solving skills.
Excellent communication skills, both written and verbal.
Detail-oriented and capable of handling multiple tasks efficiently.
Eagerness to learn and adapt to new data analysis techniques.
"""

# 5) Extract text from PDF resume
def extract_resume_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# 6) Resume heading check — handles common variations
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

# 7) Resume path — update if needed
resume_path = "/kaggle/input/resume/Pramod Saripalli Resume (1).pdf"
resume_text = extract_resume_text(resume_path)
headings, headings_warning = check_resume_headings(resume_text)

# 8) Build vector store
index, texts = build_vector_store([job_description, resume_text])

# 9) Analyze resume using Claude
def analyze_resume(job_desc, resume_text):
    context = retrieve_context(job_desc, texts, index)

    prompt = f"""
You are a job application assistant specializing in ATS optimization. Given the job description and resume, provide:
1. Job match score based on skill and experience alignment, prioritizing hard skills and relevant soft skills. Penalize the score heavily for missing critical domain-specific skills, irrelevant job role experience, or lack of industry alignment. Cap the score at 80-90 for resumes that closely match core job requirements.
2. Missing keywords or skills from the resume that are in the job description, limited to 5 high-priority terms to avoid keyword stuffing. Suggest how to integrate them contextually.

Context:
{context}

Additional Notes:
- Ensure keywords are relevant and naturally integrated, avoiding excessive repetition.
- Consider standard resume headings (Work Experience, Skills, Education) for ATS compatibility.
- If the resume lacks standard headings, note this as a potential ATS issue.
- For irrelevant job roles, ensure the score reflects significant mismatches in skills or industry experience.

Output ONLY a valid JSON object with no markdown fences or extra text:
{{
  "match_score": <int>,
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

# 10) Run and print results
if headings_warning:
    print(f"⚠️  ATS Warning: Only {len(headings)} standard headings found: {headings}")
    print("    Consider adding: Work Experience, Skills, Education, Summary, or Certifications\n")

result = analyze_resume(job_description, resume_text)
print(json.dumps(result, indent=2))
