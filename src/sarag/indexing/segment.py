"""Vietnamese word segmentation and normalisation for BM25."""

from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=1)
def _segmenter(name: str):
    if name == "pyvi":
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize
    if name == "underthesea":
        from underthesea import word_tokenize
        return lambda s: word_tokenize(s, format="text")
    raise ValueError(f"unknown segmenter {name!r}")


def segment(text: str, segmenter: str = "pyvi") -> list[str]:
    """Lower-cased word tokens; multi-syllable words are joined with '_' (e.g. 'ma_túy')."""
    text = unicodedata.normalize("NFC", text)
    seg = _segmenter(segmenter)(text).lower()
    return _TOKEN_RE.findall(seg)


def _segment_star(args):
    return segment(*args)


def segment_many(texts: list[str], segmenter: str = "pyvi", workers: int = 8, chunksize: int = 256) -> list[list[str]]:
    if workers <= 1 or len(texts) < 2000:
        return [segment(t, segmenter) for t in texts]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_segment_star, ((t, segmenter) for t in texts), chunksize=chunksize))
