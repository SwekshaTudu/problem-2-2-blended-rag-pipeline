# Kubernetes troubleshooting retrieval

Task 7: retrieve useful evidence from a large technical knowledge base using an
adaptation of IBM's Blended-RAG KNN + best_fields configuration.

Input is a troubleshooting question. Output is ranked passages with document
titles, heading paths, source URLs and component scores. This is a text retrieval
tool; it does not generate answers or require an LLM API key.

The complete corpus contains 1,611 official Kubernetes documents; 1,610 yield
17,280 searchable passages. One template-only index page produces no passage.
The included 35.2 MB index snapshot covers the full collection, not sample queries.

## Run

Requires Python 3.12. Use a virtual environment if desired. From the repository root:

```sh
python -m pip install -r requirements.txt
python src/main.py setup
python src/main.py search "How do I find previous logs from a restarting container?" --top-k 3
```

Setup verifies and restores the included index, then downloads a pinned,
approximately 24 MB quantized MiniLM encoder. Internet is required for packages
and that initial model download. Afterwards search runs locally on CPU. No GPU,
API credentials, Docker, or Elasticsearch service is required. Setup can be rerun.

For complete JSON results:

```sh
python src/main.py search "How do I debug a Kubernetes Service that has no endpoints?" --top-k 3 --json
```

The search result contains `title`, `heading`, `text`, `source_url` and `scores`.
Scores are ranking signals, not confidence probabilities. Platform-specific
documents can rank for general questions; inspect the titles and passages.

## Verify

```sh
python -m unittest discover -s tests -v
python src/check_index.py
```

The second command verifies all vectors and passage metadata, then runs real
searches. These are functional checks, not an accuracy benchmark. Exact default
setup, demonstration and test commands are also in `run.yaml`.

## Rebuild the full snapshot from source (optional)

Requires Git and internet for the documentation source. CPU embedding is the
expensive step; the observed resumed build took approximately 67 minutes on this
machine. This is why normal setup restores the complete verified snapshot.

```sh
python src/prepare_corpus.py --ref e91eff3b59c0bf813193202a4f193d37beb7efd7
python src/chunk_corpus.py
python src/encoder.py
python src/retrieval.py build --threads 4
```

Interrupted builds retain completed embedding batches. Rebuild before serving
queries from that directory. To package a completed replacement index:

```sh
python src/snapshot.py package
```

Source revisions, corpus statistics and checksums are in `data/corpus_manifest.json`
and `assets/snapshot_manifest.json`. IBM attribution, license and implementation
differences are recorded in `THIRD_PARTY_NOTICES.md`; original license texts are
retained under `licenses/`. See `docs/README.md` for the technical summary.


