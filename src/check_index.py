"""Check full snapshot integrity and execute real retrieval smoke checks."""
import hashlib
import json

import numpy as np

from retrieval import DEFAULT_INDEX, Retriever


def check():
    manifest = json.loads((DEFAULT_INDEX / "manifest.json").read_text())
    vectors = np.load(DEFAULT_INDEX / "vectors.npy", mmap_mode="r")
    assert vectors.shape == (manifest["passages"], 384), "Vector shape mismatch"
    assert np.isfinite(vectors).all(), "Non-finite vectors"
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4), "Non-normalized vectors"
    assert hashlib.sha256((DEFAULT_INDEX / "chunks.jsonl").read_bytes()).hexdigest() == manifest["chunks_sha256"], "Passage checksum mismatch"
    retriever = Retriever()
    assert len(retriever.chunks) == manifest["passages"], "Passage count mismatch"
    assert len({r["chunk_id"] for r in retriever.chunks}) == len(retriever.chunks), "Duplicate passage IDs"
    for query in ("Why is my pod stuck in CrashLoopBackOff?", "How do I view previous container logs?", "How do I debug a Service with no endpoints?"):
        result = retriever.search(query, 3)
        rows = result["results"]
        assert len(rows) == len({r["document_id"] for r in rows}) == 3, "Result diversity mismatch"
        scores = [r["scores"]["combined"] for r in rows]
        assert scores == sorted(scores, reverse=True), "Invalid ranking order"
        assert all(r["source_url"].startswith("https://github.com/kubernetes/website/blob/") for r in rows), "Missing source"
        print(f"Search OK: {query}")
    print(f"Verified {len(retriever.chunks)} passages and all vectors. Functional checks only; no accuracy claim.")


if __name__ == "__main__":
    check()
