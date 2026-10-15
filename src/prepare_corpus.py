"""Fetch an official documentation snapshot; preserve sources for later indexing."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/kubernetes/website.git"
SECTIONS = {"concepts", "tasks", "tutorials", "setup", "reference"}


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], text=True).strip()


def prepare(ref=None):
    checkout = ROOT / ".cache" / "kubernetes-website"
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if not checkout.exists():
        git("clone", "--depth", "1", "--filter=blob:none", "--sparse", REPOSITORY, checkout)
    if ref:
        if git("-C", checkout, "rev-parse", "HEAD") != ref:
            git("-C", checkout, "fetch", "--depth", "1", "origin", ref)
            git("-C", checkout, "checkout", "--detach", "FETCH_HEAD")
    git("-C", checkout, "sparse-checkout", "set", "content/en/docs")
    revision = git("-C", checkout, "rev-parse", "HEAD")
    output = ROOT / "data" / "raw"
    output.mkdir(parents=True, exist_ok=True)
    docs = checkout / "content" / "en" / "docs"
    counts = {}
    destination = output / "documents.jsonl"
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for path in sorted(docs.rglob("*.md")):
            relative = path.relative_to(docs)
            section = relative.parts[0]
            if section not in SECTIONS:
                continue
            raw = path.read_text(encoding="utf-8")
            # Preserve Markdown and Hugo markup; cleaning/chunking is a later step.
            frontmatter = re.match(r"\A---\s*\n(.*?)\n---\s*\n", raw, re.S)
            metadata = frontmatter.group(1) if frontmatter else ""
            body = raw[frontmatter.end():] if frontmatter else raw
            title_match = re.search(r"^title:\s*(.+)$", metadata, re.M)
            title = title_match.group(1).strip().strip("\"'") if title_match else path.stem
            if not body.strip():
                continue
            source_path = path.relative_to(checkout).as_posix()
            record = {
                "document_id": relative.as_posix(),
                "title": title,
                "body_markdown": body.strip(),
                "section": section,
                "source_url": f"https://github.com/kubernetes/website/blob/{revision}/{source_path}",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[section] = counts.get(section, 0) + 1
    shutil.copyfile(checkout / "LICENSE", output / "KUBERNETES_LICENSE.txt")
    manifest = {
        "repository": REPOSITORY,
        "revision": revision,
        "language": "en",
        "documents": sum(counts.values()),
        "sections": counts,
        "format": "raw Markdown, excluding YAML front matter; not yet chunked",
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
        "license": "CC-BY-4.0; see data/raw/KUBERNETES_LICENSE.txt",
    }
    manifest_path = ROOT / "data" / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Optional upstream commit SHA to reproduce a snapshot.")
    prepare(parser.parse_args().ref)
