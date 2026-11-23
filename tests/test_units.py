from sarag.indexing.units import build_units, clause_units, display_title, split_long, Unit
from sarag.parsing.parser import parse_article_level, parse_full_text

count = lambda s: len(s.split())              # whitespace "tokens" keep the tests independent of HF downloads
tokenize = lambda s: [w + " " for w in s.split()]

ARTS = [
    {"id": "1", "text": "Phạm vi điều chỉnh\n\nLuật này quy định về phòng, chống ma túy."},
    {"id": "2", "text": "Giải thích từ ngữ\n\nTrong Luật này:\n\n1. Chất ma túy là chất gây nghiện.\n\n"
                        "2. Tiền chất gồm:\n\na) Hóa chất A;\n\nb) Hóa chất B."},
]
DOC = parse_article_level("Luật Phòng, chống ma túy", ARTS)


def test_clause_units_with_breadcrumbs():
    us = list(clause_units(DOC))
    assert [u.unit_id for u in us] == ["Luật Phòng, chống ma túy|0|1|", "Luật Phòng, chống ma túy|0|2|1",
                                       "Luật Phòng, chống ma túy|0|2|2"]
    assert us[0].kind == "article" and us[1].kind == "clause"
    assert us[2].breadcrumb == "Luật Phòng, chống ma túy > Điều 2. Giải thích từ ngữ > Khoản 2"
    assert us[2].body == "2. Tiền chất gồm:\na) Hóa chất A;\nb) Hóa chất B."
    assert us[2].text.startswith(us[2].breadcrumb + "\n") and us[2].articles == ["2"]
    assert "Trong Luật này" not in us[1].text          # lead-in is added back by C3, not indexed in clauses
    assert [u.order for u in us] == [0, 1, 2]


def test_clause_units_without_breadcrumbs_keep_article_title_for_clauseless_articles():
    us = list(clause_units(DOC, breadcrumbs=False))
    assert us[0].breadcrumb == "" and us[0].text.startswith("Điều 1. Phạm vi điều chỉnh")
    assert us[1].text == "1. Chất ma túy là chất gây nghiện."


def test_article_units_b2():
    us = build_units([DOC], "article", False, count, max_tokens=1000)
    assert len(us) == 2 and us[1].text.startswith("Điều 2. Giải thích từ ngữ\nTrong Luật này:")
    assert us[1].n_tokens == count(us[1].text)


def test_fixed_units_b1_cover_articles():
    us = build_units([DOC], "fixed", False, count, 0, fixed_size=8, fixed_overlap=2, tokenize=tokenize)
    assert all(u.n_tokens <= 8 for u in us)
    assert us[0].articles == ["1"] and any(set(u.articles) == {"1", "2"} for u in us)
    words = " ".join(u.text for u in us).split()
    assert "gây" in words and "B." in words              # whole document covered


def test_split_long_respects_limit_and_marks_pieces():
    u = Unit("d|0|5|1", "d", "clause", "\n".join(f"dòng số {i} có năm từ" for i in range(30)), "Luật X > Điều 5 > Khoản 1",
             ["5"], 0, 1, 0)
    pieces = list(split_long([u], count, max_tokens=40))
    assert len(pieces) > 1 and all(p.n_tokens <= 40 for p in pieces)
    assert pieces[0].unit_id == "d|0|5|1#0" and pieces[0].breadcrumb.endswith("(phần 1/%d)" % len(pieces))
    assert "\n".join(p.body for p in pieces) == u.body
    assert all(p.articles == ["5"] and p.clause == 1 for p in pieces)


def test_split_long_cuts_a_single_huge_line():
    u = Unit("d|0|1|", "d", "article", "từ " * 500, "", ["1"])
    pieces = list(split_long([u], count, max_tokens=50))
    assert all(p.n_tokens <= 50 for p in pieces) and sum(p.n_tokens for p in pieces) == 500


def test_display_title_for_code_ids_and_lowercase_titles():
    zalo = parse_article_level("47/2011/tt-bca", [{"id": "7", "text": "1. Nội dung."}], has_titles=False)
    assert display_title(zalo) == "Thông tư 47/2011/TT-BCA"
    vle = parse_full_text("context_1", "Điều 1. Phạm vi\n\nNội dung.", title="luật trồng trọt 2018")
    assert display_title(vle) == "Luật trồng trọt 2018"


def test_fallback_document_units():
    doc = parse_full_text("c", "CÔNG VĂN\n\n1. Nội dung một.\n\n2. Nội dung hai.", title="công văn 1")
    us = list(clause_units(doc))
    assert {u.kind for u in us} == {"fallback"} and len(us) == 2
    assert us[0].breadcrumb == "Công văn 1 > Khoản 1"


def test_fixed_window_lists_majority_article_first():
    arts = [{"id": "1", "text": "Ngắn\n\nhai từ"}, {"id": "2", "text": "Dài\n\n" + "từ " * 20}]
    doc = parse_article_level("L", arts)
    us = build_units([doc], "fixed", False, count, 0, fixed_size=12, fixed_overlap=2, tokenize=tokenize)
    assert us[0].articles == ["2", "1"]             # window 0 holds 4 tokens of Điều 1 and 8 of Điều 2


def test_weighted_rrf_and_aggregation():
    from sarag.retrieval.hybrid import aggregate, rrf
    s = rrf([[1, 2], [2, 3]], k=0)
    assert s == {1: 1.0, 2: 1.5, 3: 0.5}
    assert rrf([[1, 2], [2, 1]], k=0, weights=[2.0, 1.0])[1] == 2.5
    arts = [[("d", 0, "1")], [("d", 0, "2")], [("d", 0, "2")]]
    units = [(0, 0.9), (1, 0.8), (2, 0.7)]
    assert [a for a, _ in aggregate(units, arts, "max")] == [("d", 0, "1"), ("d", 0, "2")]
    assert [a for a, _ in aggregate(units, arts, "sum_top2")] == [("d", 0, "2"), ("d", 0, "1")]
    assert [a for a, _ in aggregate(units, arts, "none")] == [("d", 0, "1"), ("d", 0, "2")]
