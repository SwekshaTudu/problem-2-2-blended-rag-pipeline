"""Frozen, quantized MiniLM encoder for CPU retrieval; no classifier preprocessing."""
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
WEIGHTS = "onnx/model_quint8_avx2.onnx"
WEIGHTS_SHA = "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21"
DEFAULT_MODEL = ROOT / "models" / "all-MiniLM-L6-v2" / REVISION


def download(directory=DEFAULT_MODEL):
    for name in (WEIGHTS, "tokenizer.json", "config.json", "sentence_bert_config.json"):
        target = Path(directory) / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}", timeout=120) as response:
                body = response.read()
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(body)
            temporary.replace(target)
    if hashlib.sha256((Path(directory) / WEIGHTS).read_bytes()).hexdigest() != WEIGHTS_SHA:
        raise ValueError("Unexpected encoder weight checksum")


class Encoder:
    def __init__(self, directory=DEFAULT_MODEL, threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = Path(directory)
        if hashlib.sha256((directory / WEIGHTS).read_bytes()).hexdigest() != WEIGHTS_SHA:
            raise ValueError("Unexpected encoder weight checksum")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        # Explicit token windows cover long passages instead of silently truncating.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / WEIGHTS), sess_options=options, providers=["CPUExecutionProvider"])
        self.names = {item.name for item in self.session.get_inputs()}

    def encode(self, texts, batch_size=32):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        windows, owners = [], []
        for owner, encoding in enumerate(self.tokenizer.encode_batch(texts, add_special_tokens=False)):
            ids = encoding.ids or [self.tokenizer.token_to_id("[UNK]")]
            for start in range(0, len(ids), 192):
                window = ids[start:start + 254]
                windows.append([101] + window + [102])
                owners.append(owner)
                if start + 254 >= len(ids):
                    break
        sums = np.zeros((len(texts), 384), dtype=np.float32)
        counts = np.zeros((len(texts), 1), dtype=np.float32)
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            length = max(map(len, batch))
            ids = np.zeros((len(batch), length), dtype=np.int64)
            mask = np.zeros_like(ids)
            for row, values in enumerate(batch):
                ids[row, :len(values)] = values
                mask[row, :len(values)] = 1
            inputs = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            tokens = self.session.run(None, {k: v for k, v in inputs.items() if k in self.names})[0]
            weights = mask[..., None].astype(np.float32)
            pooled = (tokens * weights).sum(axis=1) / weights.sum(axis=1)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
            for row, vector in enumerate(pooled):
                owner = owners[start + row]
                sums[owner] += vector
                counts[owner] += 1
        sums /= counts
        return sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-9)


if __name__ == "__main__":
    download()
    print(f"Encoder ready: {DEFAULT_MODEL}")
