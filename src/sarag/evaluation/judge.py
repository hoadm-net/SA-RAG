"""LLM judge through OpenRouter's OpenAI-compatible API, cached by prompt hash.

Not run automatically: it costs money and the judge model must be fixed in configs/models.yaml first.
"""

from __future__ import annotations

import json
import re

from ..utils.cache import JsonCache, stable_hash
from ..utils.env import require_env

JUDGE_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam, chấm điểm câu trả lời của một hệ thống hỏi đáp pháp luật.

Câu hỏi: {question}

Câu trả lời tham chiếu (của chuyên gia): {reference}

Câu trả lời cần chấm: {answer}

Chấm mức độ đúng của câu trả lời cần chấm so với câu trả lời tham chiếu, thang 1-5:
5 = đúng và đầy đủ; 4 = đúng, thiếu chi tiết nhỏ; 3 = đúng một phần; 2 = phần lớn sai; 1 = sai hoặc không trả lời.
Chỉ trả về JSON: {{"score": <1-5>, "correct": <true|false>}}"""


class Judge:
    def __init__(self, model: str, base_url: str, api_key_env: str, cache_dir, temperature: float = 0.0):
        if not model or model == "TODO":
            raise RuntimeError("Set judge.model in configs/models.yaml (one fixed model version for all runs).")
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=require_env(api_key_env))
        self.model, self.temperature = model, temperature
        self.cache = JsonCache(cache_dir, "judge")

    def score(self, question: str, reference: str, answer: str) -> dict:
        prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
        key = stable_hash({"model": self.model, "prompt": prompt})
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        result = {}
        for _ in range(2):                                   # one retry if the reply has no parsable JSON
            # Reasoning is disabled: with it on, the token cap was spent on reasoning and the content was empty.
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=256,
                messages=[{"role": "user", "content": prompt}], extra_body={"reasoning": {"enabled": False}})
            text = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.S)
            try:
                out = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                out = {}
            usage = getattr(resp, "usage", None)
            result = {"score": int(out.get("score", 0) or 0), "correct": bool(out.get("correct", False)), "raw": text,
                      "tokens": [getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)]}
            if 1 <= result["score"] <= 5:
                self.cache.set(key, result)                  # only parsed verdicts are cached
                break
        return result
