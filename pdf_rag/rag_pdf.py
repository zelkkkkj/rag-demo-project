import os
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI

PDF_PATH = "pdf_test.pdf"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 3

client = OpenAI()


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))

    return pages


def chunk_text(text, chunk_size=1200, overlap=200):
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def build_chunks(pages):
    chunks = []
    for page_num, text in pages:
        page_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(page_chunks, start=1):
            chunks.append({
                "page": page_num,
                "chunk_id": idx,
                "text": chunk
            })
    return chunks


def get_embedding(text):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32")


def embed_chunks(chunks):
    vectors = []
    for item in chunks:
        vec = get_embedding(item["text"])
        vectors.append(vec)
    return np.vstack(vectors).astype("float32")


def build_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def retrieve(question, index, chunks, k=TOP_K):
    qvec = get_embedding(question).reshape(1, -1)
    distances, indices = index.search(qvec, k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(chunks[idx])

    return results


def ask_llm(question, retrieved_chunks):
    context = "\n\n".join(
        [f"[Page {c['page']}] {c['text']}" for c in retrieved_chunks]
    )

    prompt = f"""
You are a helpful assistant.
Use the provided context if it is relevant.
If the context is not relevant, answer from general knowledge.
If you use the context, cite the page number(s) in your answer.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    resp = client.responses.create(
        model=LLM_MODEL,
        input=prompt
    )
    return resp.output_text


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}. Put the PDF in the same folder as this script."
        )

    pages = extract_text_from_pdf(PDF_PATH)

    if not pages:
        raise ValueError(
            "No text could be extracted from the PDF. If this is a scanned PDF/image PDF, you need OCR."
        )

    chunks = build_chunks(pages)

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    print(f"Extracted {len(pages)} pages and created {len(chunks)} chunks.")
    print("Embedding chunks...")

    vectors = embed_chunks(chunks)
    index = build_index(vectors)

    print("Ready. Ask a question, or type exit to stop.")

    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        retrieved = retrieve(q, index, chunks, k=TOP_K)

        print("\nRetrieved chunks:")
        for c in retrieved:
            print(f"- Page {c['page']}: {c['text'][:180]}")

        answer = ask_llm(q, retrieved)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()