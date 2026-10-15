import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from snapshot import FILES, restore


class SnapshotTests(unittest.TestCase):
    def fixture(self, root):
        assets = root / "assets"
        assets.mkdir()
        archive = assets / "index_snapshot.zip"
        bodies = {name: (name + "\n").encode() for name in FILES}
        with zipfile.ZipFile(archive, "w") as output:
            for name, body in bodies.items():
                output.writestr(name, body)
        manifest = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "files": {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()},
                    "index": {"passages": 1}}
        (assets / "snapshot_manifest.json").write_text(json.dumps(manifest))
        return archive

    def test_restore_repairs_corruption_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            self.fixture(root)
            restore(root)
            target = root / "artifacts/index/chunks.jsonl"
            target.write_text("corrupted")
            restore(root)
            self.assertEqual(target.read_text(), "chunks.jsonl\n")
            restore(root)
            self.assertEqual(target.read_text(), "chunks.jsonl\n")

    def test_corrupt_archive_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self.fixture(root)
            archive.write_bytes(b"corrupted archive")
            with self.assertRaisesRegex(ValueError, "checksum"):
                restore(root)
            self.assertFalse((root / "artifacts/index/chunks.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
