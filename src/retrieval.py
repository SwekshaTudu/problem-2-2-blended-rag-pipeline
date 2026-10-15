"""Local adaptation of IBM's KNN + best_fields query; see THIRD_PARTY_NOTICES.md."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from encoder import Encoder, DEFAULT_MODEL, MODEL_ID, REVISION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "artifacts" / "index"
TOKEN_PATTERN = r"(?u)\b\w[\w.-]*\b"


def bm25_matrix(counts, k1=1.2, b=0.75):
    counts = counts.astype(np.float32).tocsr()
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    average = max(float(lengths.mean()), 1e-9)
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((counts.shape[0] - df + 0.5) / (df + 0.5))
    row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    denominator = counts.data + k1 * (1 - b + b * lengths[row_ids] / average)
    counts.data = counts.data * (k1 + 1) / denominator * idf[counts.indices]
    return counts


def best_fields(title, content, tie_breaker=0.3):
    return np.maximum(title, content) + tie_breaker * np.minimum(title, content)


def build(chunks_path, index_dir, model_dir, threads=4):
    started = time.perf_counter()
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError("No passages to index")
    index_dir.mkdir(parents=True, exist_ok=True)
    titles = [r["title"] for r in chunks]
    contents = [r["heading"] + "\n" + r["text"] for r in chunks]
    vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN)
    vectorizer.fit(titles + contents)
    for field, texts in (("title", titles), ("content", contents)):
        sparse.save_npz(index_dir / f"{field}_bm25.npz", bm25_matrix(vectorizer.transform(texts)))
    (index_dir / "vocabulary.json").write_text(json.dumps({term: int(i) for term, i in vectorizer.vocabulary_.items()}), encoding="utf-8")
    encoder = Encoder(model_dir, threads=threads)
    checkpoint_path = index_dir / "build_progress.json"
    signature = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    completed = checkpoint.get("completed", 0) if checkpoint.get("chunks_sha256") == signature and checkpoint.get("encoder_revision") == REVISION else 0
    vectors = np.lib.format.open_memmap(index_dir / "vectors.npy", mode="r+" if completed else "w+", dtype=np.float32, shape=(len(chunks), 384))
    for start in range(completed, len(chunks), 256):
        texts = [r["title"] + "\n" + r["heading"] + "\n" + r["text"] for r in chunks[start:start + 256]]
        vectors[start:start + len(texts)] = encoder.encode(texts)
        vectors.flush()
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"completed": min(start + 256, len(chunks)), "chunks_sha256": signature,
                                        "encoder_revision": REVISION}), encoding="utf-8")
        temporary.replace(checkpoint_path)
        print(f"Embedded {min(start + 256, len(chunks))}/{len(chunks)} passages", flush=True)
    shutil.copyfile(chunks_path, index_dir / "chunks.jsonl")
    manifest = {
        "passages": len(chunks), "documents": len({r["document_id"] for r in chunks}),
        "encoder": MODEL_ID, "revision": REVISION, "dimensions": 384,
        "tie_breaker": 0.3, "dense_boost": 10.0, "dense_k": 10,
        "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "build_seconds_this_run": round(time.perf_counter() - started, 2),
        "resumed_from_passage": completed, "build_threads": threads,
        "strategy": "BM25 best_fields(title, content) + 10 * ((1 + cosine) / 2) for dense top-10 candidates",
        "adaptation": "Exact local vector search, shared lexical vocabulary and per-field BM25 statistics; no Elasticsearch stemming or ELSER. Not bit-equivalent to Elasticsearch.",
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


class Retriever:
    def __init__(self, index_dir=DEFAULT_INDEX, model_dir=DEFAULT_MODEL):
        self.directory = Path(index_dir)
        self.settings = json.loads((self.directory / "manifest.json").read_text())
        self.chunks = [json.loads(line) for line in (self.directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        vocabulary = json.loads((self.directory / "vocabulary.json").read_text())
        self.vectorizer = CountVectorizer(vocabulary=vocabulary, token_pattern=TOKEN_PATTERN, binary=True)
        self.title = sparse.load_npz(self.directory / "title_bm25.npz")
        self.content = sparse.load_npz(self.directory / "content_bm25.npz")
        self.vectors = np.load(self.directory / "vectors.npy", mmap_mode="r")
        self.encoder = Encoder(model_dir)

    def search(self, query, top_k=5):
        if not query.strip() or not 1 <= top_k <= 100:
            raise ValueError("Provide a nonempty query and top_k between 1 and 100")
        started = time.perf_counter()
        terms = self.vectorizer.transform([query]).T
        title = (self.title @ terms).toarray().ravel()
        content = (self.content @ terms).toarray().ravel()
        lexical = best_fields(title, content, self.settings["tie_breaker"])
        cosine = self.vectors @ self.encoder.encode([query])[0]
        dense_ids = np.argsort(-cosine, kind="stable")[:self.settings["dense_k"]]
        dense = np.zeros_like(cosine)
        dense[dense_ids] = (1 + cosine[dense_ids]) / 2
        scores = lexical + self.settings["dense_boost"] * dense
        ranked = np.argsort(-scores, kind="stable")
        results, used = [], set()
        # Avoid allowing a long document's overlapping passages to fill every slot.
        for i in ranked:
            row = self.chunks[i]
            if row["document_id"] in used:
                continue
            used.add(row["document_id"])
            results.append({**row, "scores": {"title_bm25": float(title[i]), "content_bm25": float(content[i]),
                            "best_fields": float(lexical[i]), "cosine": float(cosine[i]), "combined": float(scores[i])}})
            if len(results) == top_k:
                break
        return {"query": query, "retrieval_ms": round((time.perf_counter() - started) * 1000, 2), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--chunks", type=Path, default=ROOT / "data/processed/chunks.jsonl")
    builder.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    builder.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    builder.add_argument("--threads", type=int, default=4)
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    search.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.command == "build":
        build(args.chunks, args.index_dir, args.model_dir, args.threads)
    else:
        print(json.dumps(Retriever(args.index_dir, args.model_dir).search(args.query, args.top_k), indent=2))
