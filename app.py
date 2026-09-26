import os
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="💼",
    layout="wide",
)

REFUSAL_MESSAGE = (
    "I’m sorry, but I can only answer questions based on "
    "the Zyro Dynamics HR policy documents provided."
)

RAG_SYSTEM_PROMPT = """
You are the official HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.

Answer employee questions using ONLY the HR policy information in CONTEXT.

STRICT RULES:
1. Use only information contained in CONTEXT.
2. Never use general knowledge to answer the question.
3. Never invent a policy, benefit, amount, date, eligibility rule, deadline,
   approval process, or company procedure.
4. If CONTEXT does not contain enough information, use the refusal message.
5. Questions unrelated to Zyro Dynamics HR policies must be refused.
6. If multiple policy sections are relevant, combine them accurately.
7. Preserve exact numbers, percentages, durations, dates, and limits.
8. Keep the answer concise and directly answer the employee's question.
9. Do not cite information that is not present in CONTEXT.

For unsupported questions, respond exactly with:
"I’m sorry, but I can only answer questions based on the Zyro Dynamics HR policy documents provided."

CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)


def get_secret(name: str):
    """Read a secret from Streamlit Cloud secrets or environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


@st.cache_resource(show_spinner="Loading HR policy knowledge base...")
def build_rag():
    corpus_path = Path("corpus")
    pdf_files = sorted(corpus_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No HR policy PDFs were found in the corpus/ folder. "
            "Add the 11 competition HR PDFs to corpus/."
        )

    loader = PyPDFDirectoryLoader(str(corpus_path))
    documents = loader.load()

    cleaned_documents = []
    for doc in documents:
        text = doc.page_content or ""
        text = " ".join(text.split())
        if len(text) > 20:
            metadata = dict(doc.metadata)
            metadata["source_file"] = metadata.get(
                "source_file",
                Path(metadata.get("source", "Unknown")).name,
            )
            cleaned_documents.append(
                Document(page_content=text, metadata=metadata)
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(cleaned_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.65,
        },
    )

    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=700,
        api_key=api_key,
    )

    return vectorstore, retriever, llm, len(pdf_files), len(chunks)


def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown document")
        page = doc.metadata.get("page", "Unknown page")
        if isinstance(page, int):
            page += 1
        formatted.append(
            f"SOURCE {i}\n"
            f"Document: {source}\n"
            f"Page: {page}\n\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted)


def retrieve_with_scores(vectorstore, question, k=5):
    return vectorstore.similarity_search_with_score(question, k=k)


def ask_bot(question, vectorstore, retriever, llm):
    question = str(question).strip()

    if not question:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    # Guardrail: reject questions whose best vector match is too distant.
    scored = retrieve_with_scores(vectorstore, question, k=5)
    if not scored:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    best_distance = min(float(score) for _, score in scored)
    if best_distance > 1.35:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    context = format_docs(retrieved_docs)

    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": context, "question": question}
    ).strip()

    sources = []
    seen = set()

    for doc in retrieved_docs:
        source = doc.metadata.get("source_file", "Unknown document")
        page = doc.metadata.get("page", "Unknown page")
        if isinstance(page, int):
            page += 1

        key = (source, page)
        if key not in seen:
            sources.append({"document": source, "page": page})
            seen.add(key)

    return {"answer": answer, "sources": sources}


# ---------- UI ----------

st.title("💼 Zyro Dynamics HR Help Desk")
st.caption(
    "RAG-powered HR policy assistant • Answers are grounded in the provided HR documents"
)

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant retrieves relevant HR policy passages from the "
        "Zyro Dynamics document corpus and uses them as context for the answer."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Knowledge base")

    try:
        vectorstore, retriever, llm, pdf_count, chunk_count = build_rag()
        st.success("Knowledge base ready")
        st.metric("Policy PDFs", pdf_count)
        st.metric("Indexed chunks", chunk_count)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(
                        f"**{source['document']}** — page {source['page']}"
                    )

question = st.chat_input("Ask an HR policy question...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            try:
                result = ask_bot(question, vectorstore, retriever, llm)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as exc:
                answer = f"An error occurred while generating the answer: {exc}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.write(
                        f"**{source['document']}** — page {source['page']}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
