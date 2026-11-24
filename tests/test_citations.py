from sarag.generation.citations import parse_answer, parse_citations, strip_invalid, verify
from sarag.parsing.parser import parse_article_level

DOC = parse_article_level("Luật Viên chức", [
    {"id": "5", "text": "Nguyên tắc\n\n1. Tuân thủ pháp luật.\n\n2. Chịu sự giám sát."},
    {"id": "6", "text": "Quản lý\n\n1. Tuyển dụng theo tiêu chuẩn."},
])
DOCS = {"Luật Viên chức": DOC}
BLOCKS = [{"doc": "Luật Viên chức", "part": 0, "articles": ["5"], "label": "Luật Viên chức"}]


def test_parse_citations_forms():
    cs = parse_citations("A [Luật Viên chức, Điều 5, Khoản 2] B [Luật Viên chức, Điều 6] "
                         "C [Luật Viên chức, Điều 5, khoản 1, điểm a] D [không phải trích dẫn]")
    assert [(c.doc_label, c.article, c.clause, c.point) for c in cs] == [
        ("Luật Viên chức", "5", 2, None), ("Luật Viên chức", "6", None, None), ("Luật Viên chức", "5", 1, "a")]


def test_verify_statuses_and_strip():
    text = ("Đúng [Luật Viên chức, Điều 5, Khoản 2] [Luật Viên chức, Điều 6] [Luật Viên chức, Điều 5, Khoản 9] "
            "[Luật Cư trú, Điều 2] [luật viên chức, Điều 5]")
    cs = verify(parse_citations(text), BLOCKS, DOCS)
    assert [c.status for c in cs] == ["valid", "not_in_context", "not_in_tree", "unknown_document", "valid"]
    cleaned = strip_invalid(text, cs)
    assert "[Luật Viên chức, Điều 6]" not in cleaned and "[Luật Viên chức, Điều 5, Khoản 2]" in cleaned


def test_verify_accepts_shortened_label_when_unambiguous():
    blocks = [{"doc": "47/2011/tt-bca", "part": 0, "articles": ["7"], "label": "Thông tư 47/2011/TT-BCA"}]
    doc = parse_article_level("47/2011/tt-bca", [{"id": "7", "text": "1. Nội dung."}], has_titles=False)
    cs = verify(parse_citations("[47/2011/TT-BCA, Điều 7, Khoản 1]"), blocks, {"47/2011/tt-bca": doc})
    assert cs[0].status == "valid" and cs[0].doc_id == "47/2011/tt-bca"


def test_parse_answer_types():
    assert parse_answer("Đáp án: Đúng\nVì ...", "Đúng/Sai") == "Đúng"
    assert parse_answer("Đáp án: Sai.", "Đúng/Sai") == "Sai"
    assert parse_answer("Đáp án: Không đúng", "Đúng/Sai") == "Sai"
    assert parse_answer("Đáp án: C. Viên chức ...", "Trắc nghiệm") == "C"
    assert parse_answer("Đáp án: (B)", "Trắc nghiệm") == "B"
    assert parse_answer("Đáp án: 10 ngày [Luật Trọng tài thương mại, Điều 30].", "Tự luận") == "10 ngày"
    assert parse_answer("không rõ", "Trắc nghiệm") == ""


def test_token_f1_and_citation_scores():
    from sarag.datasets import Question
    from sarag.evaluation.qa import citation_scores, token_f1
    assert token_f1("10 ngày", "10 ngày") == 1.0
    assert token_f1("trong 10 ngày", "10 ngày") == 0.8
    assert token_f1("", "x") == 0.0
    q = Question("q", "?", gold_articles=[("Luật Viên chức", 0, "5")])
    pred = {"citations": [
        {"doc_id": "Luật Viên chức", "article": "5", "status": "valid"},
        {"doc_id": "Luật Viên chức", "article": "6", "status": "not_in_context"},
        {"doc_id": None, "article": "2", "status": "unknown_document"}]}
    s = citation_scores(pred, q, verified=True)
    assert (s["cit_precision"], s["cit_recall"], s["in_tree"], s["n_kept"]) == (1.0, 1.0, 2, 1)
    s = citation_scores(pred, q, verified=False)
    assert s["cit_precision"] == 0.5 and s["n_kept"] == 2


def test_parse_citations_with_title_and_point_as_clause():
    cs = parse_citations("[Luật Cư trú, Điều 27. Điều kiện đăng ký tạm trú] và [Nghị định 166/2017/nđ-cp, Điều 32, Khoản a]")
    assert [(c.doc_label, c.article, c.clause, c.point) for c in cs] == [
        ("Luật Cư trú", "27", None, None), ("Nghị định 166/2017/nđ-cp", "32", None, "a")]


def test_cluster_bootstrap_keeps_questions_together():
    from sarag.evaluation.retrieval import cluster_bootstrap_diff
    m, lo, hi = cluster_bootstrap_diff({"q1": [1.0, 1.0], "q2": [0.0, 0.0], "q3": [1.0, 0.0]})
    assert m == 0.5 and lo <= m <= hi
    m, lo, hi = cluster_bootstrap_diff({f"q{i}": [0.1] * 4 for i in range(50)})
    assert abs(m - 0.1) < 1e-12 and abs(lo - 0.1) < 1e-9 and abs(hi - 0.1) < 1e-9
