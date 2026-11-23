"""Dense index with bge-m3 (dense vectors only), exact inner-product search on GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DenseEncoder:
    def __init__(self, model_id: str = "BAAI/bge-m3", max_length: int = 512, batch_size: int = 32,
                 fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(model_id, use_fp16=fp16)
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        out = self.model.encode(texts, batch_size=self.batch_size, max_length=max_length or self.max_length,
                                return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)


class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype(np.float16)
        self._gpu = None

    def search(self, query_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Cosine top-k. Returns (unit indices, scores), each (n_queries, k)."""
        import torch

        if self._gpu is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if dev == "cuda" else torch.float32
            self._gpu = torch.from_numpy(self.vectors).to(dev, dtype)
        q = torch.from_numpy(query_vecs).to(self._gpu.device, self._gpu.dtype)
        k = min(k, self._gpu.shape[0])
        scores, idx = (q @ self._gpu.T).float().topk(k, dim=1)
        return idx.cpu().numpy(), scores.cpu().numpy()

    def save(self, path: Path) -> None:
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        return cls(np.load(path))
