"""Package or restore the complete retrieval index, verifying every file."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("chunks.jsonl", "content_bm25.npz", "title_bm25.npz", "vectors.npy", "vocabulary.json", "manifest.json")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package():
    source = ROOT / "artifacts/index"
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    archive = assets / "index_snapshot.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name in FILES:
            output.write(source / name, name)
    manifest = {
        "archive_sha256": fingerprint(archive),
        "archive_bytes": archive.stat().st_size,
        "files": {name: fingerprint(source / name) for name in FILES},
        "index": json.loads((source / "manifest.json").read_text()),
        "corpus": json.loads((ROOT / "data/corpus_manifest.json").read_text()),
    }
    (assets / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Packaged full index: {archive.stat().st_size / 1_000_000:.1f} MB")


def restore(root=ROOT):
    assets = root / "assets"
    manifest = json.loads((assets / "snapshot_manifest.json").read_text())
    archive = assets / "index_snapshot.zip"
    if fingerprint(archive) != manifest["archive_sha256"]:
        raise ValueError("Index snapshot checksum mismatch")
    target = root / "artifacts/index"
    target.mkdir(parents=True, exist_ok=True)
    if set(manifest["files"]) != set(FILES):
        raise ValueError("Unexpected snapshot manifest file list")
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != set(FILES):
            raise ValueError("Unexpected archive members")
        for name in FILES:
            destination = target / name
            expected = manifest["files"][name]
            if destination.exists() and fingerprint(destination) == expected:
                continue
            body = source.read(name)
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError(f"Snapshot file checksum mismatch: {name}")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(body)
            temporary.replace(destination)
    print(f"Verified full index: {manifest['index']['passages']} passages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["package", "restore"])
    if parser.parse_args().action == "package":
        package()
    else:
        restore()
