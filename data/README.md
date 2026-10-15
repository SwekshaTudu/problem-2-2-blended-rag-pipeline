# Kubernetes knowledge base

Source: https://github.com/kubernetes/website, English documentation under
`content/en/docs/{concepts,tasks,tutorials,setup,reference}`.

This collection supports a Kubernetes troubleshooting search tool. Documents are
retrieval evidence, not question/answer training examples. There are no intent
labels or train/validation/test splits at this stage.

From the project root, run:

```sh
python src/prepare_corpus.py
```

The first run requires Git and internet access. Later runs reuse the local
snapshot. To reproduce a specific snapshot, pass the revision recorded in
`corpus_manifest.json`:

```sh
python src/prepare_corpus.py --ref COMMIT_SHA
```

Outputs:

- `raw/documents.jsonl`: one JSON object per document, including title, raw body
  Markdown, section, stable document ID, and a source link pinned to the revision.
- `raw/KUBERNETES_LICENSE.txt`: upstream documentation license, CC BY 4.0.
- `corpus_manifest.json`: source revision, counts, size, and SHA-256 checksum.

The source is maintained by the Kubernetes documentation contributors. The
preparation script removes YAML front matter but otherwise preserves Markdown.
Hugo templates, relative links, and code examples still require processing during
index preparation. Included snippets referenced from outside these documents
have not been expanded. Raw downloads and the source checkout are ignored by Git.

## Prepare passages

```sh
python src/chunk_corpus.py
python -m unittest discover -s tests -v
```

This creates `processed/chunks.jsonl` and `processed/chunk_manifest.json`.
Each passage retains the document ID, title, heading breadcrumb, text, and pinned
source URL. Sections are split at Markdown headings, with a maximum of 180 words
and 30 words of overlap within long sections. Code examples retain their original
whitespace, although long examples can span passages. Hugo helper labels are
retained where available; template wrappers and HTML tags outside code examples
are removed. Pages containing only unsupported template includes can yield no
passages. Word limits do not guarantee encoder token limits; this must also be
handled when embedding the passages.
