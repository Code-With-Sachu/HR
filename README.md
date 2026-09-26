# Zyro Dynamics HR Help Desk — RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot for answering employee HR policy questions using the provided Zyro Dynamics HR policy PDF corpus.

## Project goal

The assistant should answer questions from the company's HR policies and refuse questions that are outside the supplied policy knowledge base.

## Architecture

```text
HR Policy PDFs
      ↓
Document Loading
      ↓
Text Cleaning
      ↓
Recursive Chunking
      ↓
HuggingFace Embeddings
      ↓
FAISS Vector Store
      ↓
MMR Retriever
      ↓
Relevance Guardrail
      ↓
Retrieved Context
      ↓
Groq LLM
      ↓
Grounded HR Answer + Sources
```

## Main components

- **PDF loading:** PyPDFDirectoryLoader
- **Chunking:** RecursiveCharacterTextSplitter
- **Chunk size:** 900 characters
- **Chunk overlap:** 150 characters
- **Embeddings:** `BAAI/bge-small-en-v1.5`
- **Vector database:** FAISS
- **Retrieval:** MMR, `k=5`, `fetch_k=20`, `lambda_mult=0.65`
- **LLM:** Groq `openai/gpt-oss-120b`
- **Temperature:** 0
- **Guardrail:** relevance threshold before generation
- **UI:** Streamlit
- **Deployment:** Streamlit Community Cloud

## Repository structure

```text
zyro-hr-rag-streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
└── corpus/
    └── <11 HR policy PDFs>
```

## Setup

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Add the 11 provided HR policy PDFs to:

```text
corpus/
```

Set your Groq key locally through Streamlit secrets:

```text
.streamlit/secrets.toml
```

Example:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Never commit the real secrets file.

Run:

```bash
streamlit run app.py
```

## Streamlit deployment

1. Push this project to a GitHub repository.
2. Put all 11 HR policy PDFs inside `corpus/`.
3. Go to Streamlit Community Cloud.
4. Create an app from the GitHub repository.
5. Select `app.py` as the entrypoint.
6. In Advanced settings → Secrets, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

7. Deploy.

## Project outcome

The Kaggle RAG pipeline generated a valid 20-row `submission.csv`. The competition submission used for the project achieved a recorded score of **76.64000 (V1)**.

## Limitations

- The assistant is intentionally restricted to the supplied HR policy corpus.
- Retrieval quality depends on the quality and completeness of the PDF text extraction.
- A relevance threshold is used as a guardrail; it may require calibration for a different corpus.
- The deployed app rebuilds the FAISS index when the Streamlit resource cache is cold.

## Responsible use

This is a policy-information assistant, not a replacement for HR decision-making. Employees should consult the official HR process when a policy question requires an action or interpretation not explicitly covered by the supplied documents.
