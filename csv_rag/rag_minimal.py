# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 00:26:22 2026

@author: 24778
"""

# rag_minimal.py
# Minimal RAG: read CSV -> embeddings -> FAISS -> retrieve -> ask LLM with context
import os
import pickle
import time
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI

# ---------- CONFIG ----------
DATA_PATH = "test_data.csv"
EMBED_CACHE = "embeddings.npz"  # 本地缓存（避免重复生成 embeddings）
DOCS_CACHE = "docs.pkl"
EMBED_MODEL = "text-embedding-3-small"
RANK_K = 3                      # 检索 top-k 段落
LLM_MODEL = "gpt-4o-mini"       # 你也可以改成 gpt-3.5-turbo 节省费用
# ----------------------------

client = OpenAI()  # 会从 OPENAI_API_KEY 环境变量读取 key

def load_docs(path):
    df = pd.read_csv(path)
    # 把要检索的字段改为你 CSV 的列名；这里假设列名是 `review`
    docs = df["review"].astype(str).tolist()
    return docs

def compute_and_cache_embeddings(docs):
    # 如果缓存存在就直接加载
    if os.path.exists(EMBED_CACHE) and os.path.exists(DOCS_CACHE):
        print("Loading embeddings from cache...")
        arr = np.load(EMBED_CACHE)["embs"]
        with open(DOCS_CACHE, "rb") as f:
            cached_docs = pickle.load(f)
        if cached_docs == docs and arr.shape[0] == len(docs):
            return arr
        else:
            print("Cache mismatch -> recomputing embeddings.")
    # 逐条生成 embedding（可以改为批量）
    embs = []
    print("Computing embeddings...")
    for i, d in enumerate(docs):
        # throttle a little if you worry about rate limits
        resp = client.embeddings.create(model=EMBED_MODEL, input=d)
        emb = np.array(resp.data[0].embedding, dtype="float32")
        embs.append(emb)
        if (i+1) % 10 == 0:
            print(f"  embedded {i+1}/{len(docs)}")
    embs = np.vstack(embs).astype("float32")
    # cache
    np.savez_compressed(EMBED_CACHE, embs=embs)
    with open(DOCS_CACHE, "wb") as f:
        pickle.dump(docs, f)
    return embs

def build_faiss_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index

def retrieve(query, index, docs, k=RANK_K):
    q_emb = np.array(client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding, dtype="float32")[None, :]
    D, I = index.search(q_emb, k)
    results = [docs[idx] for idx in I[0]]
    return results

def ask_with_context(question, context_paragraphs):
    # Compose prompt — 简洁明了
    ctx = "\n\n".join([f"- {p}" for p in context_paragraphs])
    prompt = f"""You are a helpful assistant. Use the following context (short customer reviews) to answer concisely.

Context:
{ctx}

Question:
{question}

Answer:"""
    resp = client.responses.create(model=LLM_MODEL, input=prompt)
    # try to extract text
    try:
        return resp.output_text
    except Exception:
        return str(resp)

def main():
    docs = load_docs(DATA_PATH)
    embeddings = compute_and_cache_embeddings(docs)
    index = build_faiss_index(embeddings)
    print("Index built. Ready to answer questions.\n")

    while True:
        q = input("Ask a question (or 'exit'): ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        ctx = retrieve(q, index, docs, k=RANK_K)
        print("\nRetrieved context:")
        for i, c in enumerate(ctx, 1):
            print(f"{i}. {c}")
        print("\nAsking LLM...")
        ans = ask_with_context(q, ctx)
        print("\nLLM answer:")
        print(ans)
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    main()