from sarag.expansion.structural import Candidate, Reference, expand, find_references
from sarag.indexing.store import UnitStore
from sarag.indexing.units import build_units
from sarag.packing.packer import Packer
from sarag.parsing.parser import parse_article_level

count = lambda s: len(s.split())
truncate = lambda s, n: " ".join(s.split()[:n])

ARTS = [
    {"id": "1", "text": "Phạm vi\n\nLuật này quy định về phòng, chống ma túy."},
    {"id": "2", "text": "Giải thích từ ngữ\n\nTrong Luật này, các từ ngữ dưới đây được hiểu như sau:\n\n"
                        "1. Chất ma túy là chất gây nghiện.\n\n2. Tiền chất là hóa chất.\n\n"
                        "3. Thuốc thú y có chứa các chất quy định tại các khoản 1 và 2 Điều này."},
    {"id": "3", "text": "Trách nhiệm\n\n1. Cơ quan thực hiện theo khoản 2 Điều 2 của Luật này.\n\n"
                        "2. Tổ chức tuân thủ Điều 53 Luật Dự trữ quốc gia."},
]


def make_store(variant=("clause", True)):
    doc = parse_article_level("Luật Phòng, chống ma túy", ARTS)
    units = [u.to_dict() for u in build_units([doc], variant[0], variant[1], count, 1000)]
    return UnitStore(units, [doc])


# ----------------------------------------------------------------------------- cross-references

def test_find_references_internal_and_external():
    text = ("theo khoản 2 Điều 2 của Luật này; các khoản 4, 5 và 7 Điều này; điểm a khoản 1 Điều 5; "
            "Điều 53 Luật Dự trữ quốc gia; khoản 3 Điều 30 Nghị định số 29/2012/NĐ-CP; Điều 49, Điều 50 của Bộ luật này")
    refs = find_references(text)
    assert Reference("2", (2,)) in refs
    assert Reference("", (4, 5, 7)) in refs
    assert Reference("5", (1,)) in refs                     # bare reference: same document
    assert Reference("49", ()) in refs and Reference("50", ()) in refs
    assert all(r.article not in ("53", "30") for r in refs)  # other documents are skipped


def test_expand_adds_referenced_clauses_with_decay():
    store = make_store()
    src = store.by_id["Luật Phòng, chống ma túy|0|3|1"]
    out = expand([Candidate(src, 1.0)], store, decay=0.5)
    target = store.by_id["Luật Phòng, chống ma túy|0|2|2"]
    assert [(c.unit, c.score, c.source) for c in out] == [(src, 1.0, "retrieved"), (target, 0.5, "xref")]


def test_expand_this_article_and_never_downgrades_retrieved():
    store = make_store()
    src = store.by_id["Luật Phòng, chống ma túy|0|2|3"]
    k1 = store.by_id["Luật Phòng, chống ma túy|0|2|1"]
    out = {c.unit: c for c in expand([Candidate(src, 1.0), Candidate(k1, 0.9)], store)}
    assert out[k1].score == 0.9 and out[k1].source == "retrieved"
    assert out[store.by_id["Luật Phòng, chống ma túy|0|2|2"]].source == "xref"


# ----------------------------------------------------------------------------- packing

def test_render_groups_by_article_with_lead_in_and_original_order():
    store = make_store()
    ids = [store.by_id[f"Luật Phòng, chống ma túy|0|2|{k}"] for k in (3, 1)]
    text, meta = Packer(store, count).render(ids, {ids[0]: 1.0, ids[1]: 0.5})
    assert text.startswith("[Luật Phòng, chống ma túy, Điều 2. Giải thích từ ngữ]\n"
                           "Trong Luật này, các từ ngữ dưới đây được hiểu như sau:\n1. Chất ma túy")
    assert text.index("1. Chất") < text.index("3. Thuốc")
    assert meta[0]["articles"] == ["2"]


def test_render_without_lead_in():
    store = make_store()
    i = store.by_id["Luật Phòng, chống ma túy|0|2|1"]
    text, _ = Packer(store, count, lead_in=False).render([i], {i: 1.0})
    assert text == "[Luật Phòng, chống ma túy, Điều 2. Giải thích từ ngữ]\n1. Chất ma túy là chất gây nghiện."


def test_greedy_packing_respects_budget():
    store = make_store()
    cands = [Candidate(i, 1.0 / (1 + i)) for i in range(len(store.units))]
    for budget in (10, 25, 40, 1000):
        p = Packer(store, count).pack(cands, budget)
        assert p.n_tokens <= budget
    assert len(Packer(store, count).pack(cands, 1000).units) == len(store.units)


def test_score_per_token_prefers_dense_units():
    store = make_store()
    short = store.by_id["Luật Phòng, chống ma túy|0|2|2"]      # few tokens
    long_ = store.by_id["Luật Phòng, chống ma túy|0|2|3"]      # more tokens, slightly higher score
    budget = count(Packer(store, count).render([short], {short: 1})[0])
    p = Packer(store, count).pack([Candidate(long_, 1.0), Candidate(short, 0.9)], budget)
    assert p.units == [short]


def test_topk_truncate_cuts_last_unit_to_fill_budget():
    store = make_store(("article", False))
    cands = [Candidate(store.by_id[f"Luật Phòng, chống ma túy|0|{n}|"], s) for n, s in (("2", 1.0), ("3", 0.9))]
    full = Packer(store, count, truncate=truncate).pack(cands, 10_000, "topk_truncate")
    assert full.truncated_unit is None and len(full.units) == 2
    budget = full.n_tokens - 5
    p = Packer(store, count, truncate=truncate, min_tail=1).pack(cands, budget, "topk_truncate")
    assert p.n_tokens <= budget and p.truncated_unit == cands[1].unit
    assert p.text.count("[Luật Phòng, chống ma túy, Điều") == 2


def test_b2_article_unit_title_not_duplicated():
    store = make_store(("article", False))
    i = store.by_id["Luật Phòng, chống ma túy|0|1|"]
    text, _ = Packer(store, count).render([i], {i: 1.0})
    assert text == "[Luật Phòng, chống ma túy, Điều 1. Phạm vi]\nLuật này quy định về phòng, chống ma túy."
