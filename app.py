#importing required libraries
import os
import json
import PyPDF2
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai

# Configuring Gemini API
try:
    user_secrets = UserSecretsClient()
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
except:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyALyqhdzcO8AamcfeQUFBCwkhn1uUhCf2E"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# Embeddings for RAG
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# extract text from PDF resume
def extract_resume_text(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

#  Analyze Resume and Job Description Using Gemini Model
def analyze_resume(job_desc, resume_text):
    # Store documents in FAISS vector store for RAG
    documents = [
        Document(page_content=job_desc, metadata={"type": "job_description"}),
        Document(page_content=resume_text, metadata={"type": "resume"})
    ]
    vector_store = FAISS.from_documents(documents, embeddings)

    # Retrieve relevant documents using RAG
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(job_desc)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Generate ATS score and missing keywords
    prompt = f"""
    You are a job application assistant specializing in ATS optimization. Given the job description and resume, provide:
    1. Job match score based on skill and experience alignment, prioritizing hard skills and relevant soft skills. Penalize the score heavily for missing critical domain-specific skills, irrelevant job role experience, or lack of industry alignment. Cap the score at 90-100 for resumes that closely match core job requirements.
    2. Missing keywords or skills from the resume that are in the job description, limited to 5 high-priority terms to avoid keyword stuffing. Suggest how to integrate them contextually.
    
    Context:
    {context}

    Output in JSON format:
    ```json
    {{
      "match_score": <int>,
      "missing_keywords": [{{"keyword": "<string>", "suggestion": "<string>"}}]
    }}
    ```
    """

    # Gemini API
    response = model.generate_content(prompt)
    try:
        # JSON from response
        json_output = json.loads(response.text.strip("```json\n").strip("\n```"))
        return json_output
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON output"}

# Streamlit UI
st.title("ATS Resume Optimizer")
st.text("Check your resume ATS score based on the job you are applying and optimize it")

# Upload file and input job description
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
job_description = st.text_area("Enter the Job Description", height=150)

if st.button("Submit"):
    if uploaded_file is not None and job_description:
        with st.spinner("Analyzing..."):
            resume_text = extract_resume_text(uploaded_file)
            result = analyze_resume(job_description, resume_text)

        # Display results
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("ATS Score")
            st.write(f" Match Score: {result['match_score']}")
            st.subheader("Missing Keywords & Suggestions: ")
            for kw in result.get("missing_keywords", []):
                st.write(f"- **{kw['keyword']}**: {kw['suggestion']}")
    else:
        st.warning("Please upload a resume and enter a job description.")
