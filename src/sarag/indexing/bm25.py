"""BM25 index over segmented unit texts (bm25s)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .segment import segment_many


class BM25Index:
    def __init__(self, retriever, segmenter: str = "pyvi"):
        self.retriever = retriever
        self.segmenter = segmenter

    @classmethod
    def build(cls, texts: list[str], segmenter: str = "pyvi", k1: float = 1.2, b: float = 0.75,
              workers: int = 8) -> "BM25Index":
        import bm25s

        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(segment_many(texts, segmenter, workers), show_progress=False)
        return cls(retriever, segmenter)

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (unit indices, scores), each of shape (n_queries, k). Unknown words score 0."""
        k = min(k, int(self.retriever.scores["num_docs"]))
        res, scores = self.retriever.retrieve(segment_many(queries, self.segmenter, workers=1), k=k,
                                              show_progress=False)
        return np.asarray(res), np.asarray(scores)

    def save(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(path))
        (Path(path) / "segmenter.txt").write_text(self.segmenter)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        import bm25s

        seg = (Path(path) / "segmenter.txt").read_text().strip()
        return cls(bm25s.BM25.load(str(path)), seg)
