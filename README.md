# ATS Resume Optimizer

An AI-powered resume analyzer that scores your resume against a job description and provides honest feedback on missing keywords and areas for improvement.

## Demo

[Live App on Streamlit](https://resumeatswithclaudeapi-7gdjzm2bkdnipcpzbgga65.streamlit.app/))

## What It Does

- Uploads a resume in PDF format
- Matches it against a pasted job description
- Returns an ATS match score (0-100)
- Identifies up to 5 missing high-priority keywords with suggestions on where to add them
- Gives honest, balanced feedback on resume strengths and weaknesses

## Tech Stack

| Tool | Purpose |
|------|---------|
| Anthropic Claude Haiku | ATS analysis and feedback generation |
| FAISS | Vector store for document retrieval |
| SentenceTransformers (all-MiniLM-L6-v2) | Text embeddings for semantic search |
| pypdf | PDF text extraction |
| Streamlit | Web interface |

## How It Works

1. Resume and job description are converted into vector embeddings using SentenceTransformers
2. FAISS retrieves the most relevant context from both documents
3. Claude Haiku analyzes the context and returns a structured JSON response with a match score, feedback, and missing keywords
4. Results are displayed in the Streamlit UI


## Limitations

- Creative or heavily designed resume layouts may not extract cleanly from PDF
- Tool prioritizes keyword matches — context alignment may not always be perfect
- Scores above 80 are only given when the resume genuinely competes for the role

## Future Enhancements

- Keyword context checking using BERT to avoid false positives
- Industry-specific scoring for tech, finance, healthcare etc.
- Conversational chatbot for instant resume improvement tips
- Automatic resume bullet point rewriter based on missing keywords

[Blog Post](https://ats-resume-optimizer.blogspot.com/2025/04/building-ats-optimized-resume-analyzer.html)
