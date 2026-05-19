# RAG Demo Project

A beginner-friendly Retrieval-Augmented Generation (RAG) project built with:

- OpenAI API
- FAISS vector search
- CSV-based retrieval
- PDF-based retrieval

This project demonstrates how RAG systems work by:
1. Converting documents into embeddings
2. Retrieving the most relevant chunks
3. Using an LLM to generate answers from retrieved context

---

# Features

## CSV RAG
- Semantic search over CSV data
- Embedding-based retrieval
- Context-aware question answering

## PDF RAG
- PDF text extraction
- Chunking with overlap
- Retrieval over long documents

---

# Project Structure

```text
Rag_Demo_20260316/
├── csv_rag/
│   ├── rag_minimal.py
│   └── test_data.csv
│
├── pdf_rag/
│   ├── rag_pdf.py
│   └── pdf_test.pdf
│
├── requirements.txt
├── README.md
└── .env.example
