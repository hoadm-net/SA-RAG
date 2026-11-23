from sarag.parsing.parser import parse_article_body, parse_article_level, parse_full_text
from sarag.parsing.patterns import normalize_text
from sarag.parsing.tree import Article, Document

ALQAC_ARTICLE = (
    "Chế độ đào tạo, bồi dưỡng viên chức\n\n"
    "1. Việc đào tạo, bồi dưỡng được thực hiện đối với viên chức.\n\n"
    "2. Nội dung, chương trình phải căn cứ vào tiêu chuẩn.\n\n"
    "3. Hình thức đào tạo, bồi dưỡng viên chức gồm:\n\n"
    "a) Đào tạo, bồi dưỡng theo tiêu chuẩn chức vụ quản lý;\n\n"
    "b) Bồi dưỡng theo tiêu chuẩn chức danh nghề nghiệp."
)


# ----------------------------------------------------------------------------- article body

def test_alqac_article_title_clauses_points():
    doc = parse_article_level("Luật Viên chức", [{"id": "33", "text": ALQAC_ARTICLE}])
    a = doc.article("33")
    assert a.title == "Chế độ đào tạo, bồi dưỡng viên chức"
    assert [c.number for c in a.clauses] == [1, 2, 3]
    assert [p.label for p in a.clause(3).points] == ["a", "b"]
    assert a.clause(3).text == "Hình thức đào tạo, bồi dưỡng viên chức gồm:"
    assert a.rejected_lines == 0


def test_article_without_title_starts_with_clause():
    doc = parse_article_level("Hiến pháp", [{"id": "2", "text": "1. Nhà nước pháp quyền.\n\n2. Quyền lực nhà nước."}])
    a = doc.article("2")
    assert a.title == "" and [c.number for c in a.clauses] == [1, 2]


def test_zalo_articles_have_no_title_line():
    text = "Thông tư này hướng dẫn tuần tra, canh gác bảo vệ đê."
    doc = parse_article_level("01/2009/tt-bnn", [{"id": "1", "text": text}], has_titles=False)
    a = doc.article("1")
    assert a.title == "" and a.lead_in == text and not a.clauses


def test_lead_in_before_clause_one():
    a = parse_article_body(["Trong Luật này, các từ ngữ dưới đây được hiểu như sau:", "1. Chất ma túy là ...",
                            "2. Tiền chất là ..."], Article("2"))
    assert a.lead_in.startswith("Trong Luật này") and len(a.clauses) == 2


def test_out_of_sequence_numbers_are_continuation():
    lines = ["1. Mức phạt như sau:", "2020. Năm áp dụng.", "2. Khoản hai.", "5. Không phải khoản năm."]
    a = parse_article_body(lines, Article("9"))
    assert [c.number for c in a.clauses] == [1, 2]
    assert "2020. Năm áp dụng." in a.clause(1).text
    assert "5. Không phải khoản năm." in a.clause(2).text
    assert a.rejected_lines == 1        # "2020." never matches the clause pattern (max 3 digits)


def test_point_sequence_uses_vietnamese_alphabet():
    lines = ["1. Gồm:", "a) A;", "b) B;", "c) C;", "d) D;", "đ) Đ;", "e) E;", "g) G.", "x) không theo thứ tự"]
    a = parse_article_body(lines, Article("1"))
    assert [p.label for p in a.clause(1).points] == ["a", "b", "c", "d", "đ", "e", "g"]
    assert a.clause(1).points[-1].text.endswith("x) không theo thứ tự")


def test_points_directly_under_article():
    a = parse_article_body(["Đối tượng đánh giá gồm:", "a) Chương trình;", "b) Học viên."], Article("1"))
    assert not a.clauses and [p.label for p in a.points] == ["a", "b"]


def test_bullets_and_subclauses_attach_to_current_element():
    a = parse_article_body(["1. Gồm:", "a) Điểm a:", "- gạch đầu dòng", "1.1. mục con"], Article("1"))
    assert a.clause(1).points[0].text == "Điểm a:\n- gạch đầu dòng\n1.1. mục con"


# ----------------------------------------------------------------------------- full text

RAW = (
    "BỘ NỘI VỤ\r\n\n  -------\n\nTHÔNG TƯ\n\nCăn cứ khoản 5, Điều 30 Nghị định số\r\n\n101/2017/NĐ-CP;\n\n"
    "Chương I\n\nQUY\r\n\nĐỊNH CHUNG\n\n"
    "Điều 1. Phạm vi điều\r\n\nchỉnh và đối tượng đánh giá\n\n"
    "1. Thông tư này quy định về đánh giá\r\n\nchất lượng bồi dưỡng.\n\n"
    "2. Đối tượng đánh giá gồm:\n\na) Chương trình;\n\nb) Học viên.\n\n"
    "Chương II\n\nTỔ CHỨC THỰC HIỆN\n\n"
    "Mục 1. TRÁCH NHIỆM\n\n"
    "Điều 2. Hiệu lực thi hành\n\nThông tư này có hiệu lực từ ngày ký.\n\n"
    "“Điều 1. Điều được trích dẫn trong văn bản sửa đổi.”\n\n"
    "Nơi nhận:\n\n- Như trên;\n\nBỘ TRƯỞNG\n\nNguyễn Văn A"
)


def test_normalize_joins_soft_wraps():
    assert normalize_text("Phạm vi điều\r\n\nchỉnh\n\nĐiều 2") == "Phạm vi điều chỉnh\nĐiều 2"


def test_full_text_raw_document():
    doc = parse_full_text("context_1", RAW, title="thông tư 1")
    assert [a.number for a in doc.articles] == ["1", "2"]
    a1, a2 = doc.articles
    assert a1.title == "Phạm vi điều chỉnh và đối tượng đánh giá"
    assert a1.chapter == "I" and doc.chapters["I"] == "QUY ĐỊNH CHUNG"
    assert a1.clause(1).text == "Thông tư này quy định về đánh giá chất lượng bồi dưỡng."
    assert [p.label for p in a1.clause(2).points] == ["a", "b"]
    assert a2.chapter == "II" and a2.section == "1"
    # quoted article kept as text; signature block cut
    assert "trích dẫn" in a2.lead_in
    assert "Nơi nhận" not in a2.full_text() and "BỘ TRƯỞNG" not in a2.full_text()
    assert "Căn cứ khoản 5" in doc.preamble and not doc.fallback


def test_full_text_tagged_document():
    text = ("LUẬT\nTRỒNG TRỌT\n\nChương I. NHỮNG QUY ĐỊNH CHUNG\n\nChương II. GIỐNG CÂY TRỒNG\n\n"
            "Mục 1. NGUỒN GEN\nChương: II\n\n"
            "Điều 1. Phạm vi điều chỉnh\nLuật này quy định về hoạt động trồng trọt.\nChương: I\n\n"
            "Điều 13. Yêu cầu chung\n1. Giống cây trồng chính.\n2. Giống khác.\nMục: 2\nChương: II\n")
    doc = parse_full_text("context_2", text)
    a1, a13 = doc.articles
    assert (a1.chapter, a13.chapter, a13.section) == ("I", "II", "2")
    assert doc.chapters == {"I": "NHỮNG QUY ĐỊNH CHUNG", "II": "GIỐNG CÂY TRỒNG"}
    assert "Chương:" not in a13.full_text() and [c.number for c in a13.clauses] == [1, 2]


def test_inline_article_content_after_dash():
    doc = parse_full_text("c", "QUYẾT ĐỊNH:\n\nĐiều 1. - Sửa đổi điều 19 như sau:\n\nĐiều 2. - Mẫu bằng tốt nghiệp.")
    a1 = doc.article("1")
    assert a1.title == "" and a1.lead_in == "Sửa đổi điều 19 như sau:"


def test_document_without_articles_falls_back():
    doc = parse_full_text("c", "CÔNG VĂN\n\nVề việc hướng dẫn.\n\n1. Nội dung một.\n\n2. Nội dung hai.")
    assert doc.fallback and len(doc.articles) == 1
    assert doc.articles[0].number == "" and [c.number for c in doc.articles[0].clauses] == [1, 2]


def test_roundtrip_serialisation():
    doc = parse_full_text("context_1", RAW)
    again = Document.from_dict(doc.to_dict())
    assert again == doc


ATTACHED = (
    "QUYẾT ĐỊNH:\n\nĐiều 1. Ban hành kèm theo Quyết định này Quy định về bồi thường.\n\n"
    "Điều 2. Quyết định có hiệu lực từ ngày ký.\n\n"
    "Điều 3. Các ông (bà) căn cứ Quyết định thi hành./.\n\n"
    "TM. ỦY BAN NHÂN DÂN\n\nCHỦ TỊCH\n\nNguyễn Văn B\n\n"
    "QUY ĐỊNH\n\nVỀ BỒI THƯỜNG\n\n(Ban hành kèm theo Quyết định số 3000/2017/QĐ-UBND)\n\n"
    "Chương I\n\nQUY ĐỊNH CHUNG\n\n"
    "Điều 1. Phạm vi điều chỉnh\n\nQuy định này quy định về bồi thường.\n\n"
    "Điều 2. Đối tượng áp dụng\n\n1. Cơ quan nhà nước.\n\n2. Người sử dụng đất."
)


def test_attached_regulation_restarts_numbering_as_new_part():
    doc = parse_full_text("c", ATTACHED)
    assert [(a.part, a.number) for a in doc.articles] == [(0, "1"), (0, "2"), (0, "3"), (1, "1"), (1, "2")]
    assert "TM." not in doc.article("3").full_text() and "QUY ĐỊNH" not in doc.article("3").full_text()
    assert doc.article("2").title == "Đối tượng áp dụng"            # attached regulation preferred
    assert doc.article("2", part=0).title == "Quyết định có hiệu lực từ ngày ký."
    assert doc.article("1").chapter == "I"


def test_quoted_restart_without_signature_is_not_a_new_part():
    text = "Điều 1. Sửa đổi\n\n1. Sửa đổi Điều 22 như sau:\n\nĐiều 1. Không phải điều mới.\n\nĐiều 2. Hiệu lực"
    doc = parse_full_text("c", text)
    assert [a.number for a in doc.articles] == ["1", "2"] and {a.part for a in doc.articles} == {0}


def test_article_level_last_article_drops_annex():
    arts = [{"id": "1", "text": "1. Khoản một."},
            {"id": "2", "text": "Bộ trưởng chịu trách nhiệm thi hành.\nPHỤ LỤC I\nMẪU VĂN BẢN\n1. Mục một."}]
    doc = parse_article_level("58/2021/nđ-cp", arts, has_titles=False)
    a = doc.article("2")
    assert a.full_text() == "Bộ trưởng chịu trách nhiệm thi hành." and not a.clauses


def test_lookup_prefers_main_part_over_annexed_forms():
    text = ("Điều 1. Phạm vi\n\n1. Nội dung dài của nghị định về phạm vi điều chỉnh.\n\n"
            "Điều 2. Hiệu lực\n\n1. Có hiệu lực từ ngày ký.\n\n2. Áp dụng toàn quốc.\n\n"
            "TM. CHÍNH PHỦ\n\nMẪU SỐ 01\n\nĐiều 1. ...\n\nĐiều 2. ...")
    doc = parse_full_text("c", text)
    assert {a.part for a in doc.articles} == {0, 1}
    assert doc.article("2").part == 0 and doc.article("2").clause(2) is not None


def test_lookup_keeps_long_main_text_over_annexed_charters():
    main = "".join(f"Điều {i}. Điều {i}\n\n1. Nội dung ngắn.\n\n" for i in range(1, 7))
    charter = "".join(f"Điều {i}. Mẫu\n\n1. {'Nội dung mẫu điều lệ rất dài. ' * 20}\n\n" for i in range(1, 7))
    doc = parse_full_text("c", main + "BỘ TRƯỞNG\n\n" + charter)
    assert doc.main_part() == 0 and doc.article("2").part == 0
