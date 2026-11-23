from pathlib import Path

from sarag.utils.cache import JsonCache, stable_hash
from sarag.utils.config import load_config
from sarag.utils.io import done_ids, read_jsonl, write_jsonl


def test_sarag_config_inherits_defaults():
    cfg = load_config("configs/experiments/sarag.yaml")
    assert cfg["name"] == "sarag"
    assert cfg["packing"]["budget"] == 2048
    assert cfg["retrieval"]["aggregation"] == "max"


def test_ablation_overrides_only_its_component():
    cfg = load_config("configs/experiments/ablation_no_packing.yaml")
    assert cfg["packing"]["budget"] == 2048 and cfg["packing"]["strategy"] == "topk_truncate"
    assert cfg["expansion"]["enabled"] is True


def test_cli_overrides():
    cfg = load_config("configs/experiments/sarag.yaml", ["packing.budget=1024", "generation.generator=qwen3-1.7b"])
    assert cfg["packing"]["budget"] == 1024
    assert cfg["generation"]["generator"] == "qwen3-1.7b"


def test_jsonl_roundtrip_and_resume(tmp_path: Path):
    p = tmp_path / "pred.jsonl"
    write_jsonl(p, [{"question_id": "q1", "answer": "Điều 5"}])
    write_jsonl(p, [{"question_id": "q2"}], append=True)
    assert [r["question_id"] for r in read_jsonl(p)] == ["q1", "q2"]
    assert done_ids(p) == {"q1", "q2"}


def test_cache(tmp_path: Path):
    c = JsonCache(tmp_path, "judge")
    k = stable_hash({"prompt": "xin chào"})
    assert c.get(k) is None
    c.set(k, {"score": 5})
    assert c.get(k) == {"score": 5}
