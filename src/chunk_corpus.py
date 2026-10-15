"""Create heading-aware passages from the downloaded Kubernetes documentation."""

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shortcode_text(match):
    value = match.group(0)
    # Keep visible labels embedded inside Hugo helpers, rather than losing them.
    label = re.search(r'\b(?:text|title)\s*=\s*([\"\'])(.*?)\1', value, re.S)
    return label.group(2) if label else ""


def sections(markdown):
    """Yield breadcrumb/text pairs; headings in fenced examples are not sections."""
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.S)
    # Replace multi-line helpers outside fenced examples before parsing headings.
    prepared = []
    outside = []
    code_fence = None
    for line in markdown.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker and code_fence is None:
            prepared.append(re.sub(r"\{\{[<%].*?[%>]\}\}", shortcode_text, "".join(outside), flags=re.S))
            outside = []
            code_fence = marker.group(1)
            prepared.append(line)
        elif code_fence is not None:
            prepared.append(line)
            if marker and marker.group(1)[0] == code_fence[0] and len(marker.group(1)) >= len(code_fence):
                code_fence = None
        else:
            outside.append(line)
    prepared.append(re.sub(r"\{\{[<%].*?[%>]\}\}", shortcode_text, "".join(outside), flags=re.S))
    markdown = "".join(prepared)
    headings = []
    lines = []
    fence = None
    for line in markdown.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append(line)
            continue
        if fence is not None:
            lines.append(line)
            continue
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            if "\n".join(lines).strip():
                yield " > ".join(text for _, text in headings), "\n".join(lines).strip()
            lines = []
            level = len(heading.group(1))
            text = re.sub(r"\s*\{#.*?\}\s*$", "", heading.group(2))
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, text))
            continue
        line = re.sub(r"\{\{[<%].*?[%>]\}\}", shortcode_text, line)
        # Keep link labels; citations come from the preserved source URL.
        line = re.sub(r"!?\[([^\]]*)\]\([^\n]*?\)", r"\1", line)
        line = re.sub(r"</?[A-Za-z][^>]*>", "", line)
        lines.append(html.unescape(line))
    if "\n".join(lines).strip():
        yield " > ".join(text for _, text in headings), "\n".join(lines).strip()


def passages(text, max_words, overlap):
    """Bound passage length while preserving whitespace and overlapping context."""
    words = list(re.finditer(r"\S+", text))
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        yield text[words[start].start():words[end - 1].end()]
        if end == len(words):
            break
        start = end - overlap


def chunk_documents(documents, max_words=180, overlap=30):
    if max_words < 1 or not 0 <= overlap < max_words:
        raise ValueError("Require max_words > 0 and 0 <= overlap < max_words")
    for document in documents:
        number = 0
        for heading, body in sections(document["body_markdown"]):
            for text in passages(body, max_words, overlap):
                yield {
                    "chunk_id": f"{document['document_id']}::passage-{number:05d}",
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "heading": heading,
                    "text": text,
                    "section": document["section"],
                    "source_url": document["source_url"],
                    "word_count": len(text.split()),
                }
                number += 1


def prepare(input_path, output_path, max_words, overlap):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    count = 0
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8", newline="\n") as output:
        documents = (json.loads(line) for line in source if line.strip())
        for chunk in chunk_documents(documents, max_words, overlap):
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            seen.add(chunk["document_id"])
            count += 1
    manifest = {
        "documents_with_passages": len(seen),
        "passages": count,
        "max_words": max_words,
        "overlap_words": overlap,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "limitations": [
            "Word limits are not encoder token limits; embedding input must also be token-budgeted.",
            "External Hugo includes are not expanded; unlabeled template helpers are removed.",
            "Long sections including code or tables can span multiple overlapping passages.",
        ],
    }
    (output_path.parent / "chunk_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/documents.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/chunks.jsonl")
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=30)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    prepare(args.input, args.output, args.max_words, args.overlap)
