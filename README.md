# ATS Resume Optimizer

An AI-powered resume analyzer that scores your resume against a job description and provides honest feedback on missing keywords and areas for improvement.

## Demo

[Live App on Streamlit](https://share.streamlit.io) <!-- replace with your actual link -->

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

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/PramodSaripalli/Resume_ATS_with_Google_Flash
cd Resume_ATS_with_Google_Flash
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.streamlit/secrets.toml` file:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Get your API key at [console.anthropic.com](https://console.anthropic.com)

### 4. Run the app

```bash
streamlit run app.py
```

## Requirements

```
anthropic
pypdf
sentence-transformers
faiss-cpu
streamlit
```

## Project Structure

```
├── app.py               # Streamlit app
├── requirements.txt     # Dependencies
└── README.md
```

## Limitations

- Creative or heavily designed resume layouts may not extract cleanly from PDF
- Tool prioritizes keyword matches — context alignment may not always be perfect
- Scores above 80 are only given when the resume genuinely competes for the role

## Future Enhancements

- Keyword context checking using BERT to avoid false positives
- Industry-specific scoring for tech, finance, healthcare etc.
- Conversational chatbot for instant resume improvement tips
- Automatic resume bullet point rewriter based on missing keywords

## Author

**Pramod Saripalli**

[Blog Post](https://your-blog-link) <!-- replace with your actual link --> | [LinkedIn](https://linkedin.com/in/pramod-saripalli)
