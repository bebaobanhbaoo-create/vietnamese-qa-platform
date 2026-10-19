from huflit_qa.config import ChunkingSettings
from huflit_qa.processing.chunker import index_text
from huflit_qa.processing.chunker_v2 import _pack, _split_to_fit, chunk_quy_che, chunk_thong_bao

CFG = ChunkingSettings(strategy="v2", max_words=30, min_words=5)


def _record(content, doc_type="Quy chế"):
    return {
        "id": "7_1",
        "doc_title": "Quy chế đào tạo",
        "doc_type": doc_type,
        "doc_url": "https://portal.huflit.edu.vn/News/Detail/7/x",
        "date_posted": "2024-01-01",
        "content": content,
    }


REGULATION = """QUY CHẾ ĐÀO TẠO

Chương I
NHỮNG QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Quy chế này áp dụng cho sinh viên đại học chính quy.

Chương II
ĐÁNH GIÁ KẾT QUẢ HỌC TẬP

Điều 9. Đánh giá học phần
1. Mỗi học phần có tối thiểu hai điểm thành phần được chấm theo thang điểm mười trên hệ thống.
2. Sinh viên vắng thi không có lý do chính đáng phải nhận điểm không cho lần thi đó.
3. Sinh viên vắng thi có lý do chính đáng được dự thi ở một đợt khác và tính điểm lần đầu."""


def test_chunks_respect_word_budget_and_keep_khoan_numbers():
    chunks = chunk_quy_che(_record(REGULATION), CFG)
    assert all(len(c["text"].split()) <= CFG.max_words * 1.25 for c in chunks)
    dieu9 = [c for c in chunks if c["chunk_title"] == "Điều 9. Đánh giá học phần"]
    assert len(dieu9) >= 2  # 3 Khoản of ~17 words do not fit in 30 words
    assert any(c["text"].lstrip().startswith("2.") for c in dieu9)


def test_context_carries_chapter_and_article():
    chunks = chunk_quy_che(_record(REGULATION), CFG)
    dieu9 = next(c for c in chunks if c["chunk_title"] == "Điều 9. Đánh giá học phần")
    assert (
        dieu9["context"] == "Quy chế đào tạo > Chương II ĐÁNH GIÁ KẾT QUẢ HỌC TẬP > Điều 9. Đánh giá học phần"
    )
    assert index_text(dieu9, context_prefix=True).startswith(dieu9["context"] + "\n\n")
    assert index_text(dieu9, context_prefix=False) == dieu9["text"]


def test_chapter_heading_is_not_left_in_previous_article():
    chunks = chunk_quy_che(_record(REGULATION), CFG)
    dieu1 = next(c for c in chunks if c["chunk_title"] == "Điều 1. Phạm vi điều chỉnh")
    assert "Chương II" not in dieu1["text"]


def test_split_falls_back_to_sentences_then_words():
    sentence = "Câu này có đúng mười từ để kiểm tra việc tách câu."
    parts = _split_to_fit(" ".join([sentence] * 5), max_words=25)
    assert all(len(p.split()) <= 25 for p in parts)
    assert all(p.endswith(".") for p in parts)
    long_word_run = " ".join(["từ"] * 70)
    assert [len(p.split()) for p in _split_to_fit(long_word_run, 30)] == [30, 30, 10]


def test_pack_merges_short_tail():
    assert _pack(["a " * 20, "b " * 3], max_words=20, min_words=5) == [("a " * 20) + "\n" + ("b " * 3)]


def test_announcement_paragraphs_are_packed():
    paragraphs = "\n\n".join(f"Đoạn {i} có vài từ ngắn." for i in range(4))  # 4 x 6 words fits in 30
    chunks = chunk_thong_bao(_record(paragraphs, doc_type="Thông báo"), CFG)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "7_1_c0"
    assert chunks[0]["doc_type"] == "thong_bao"
