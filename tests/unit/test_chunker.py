from huflit_qa.config import ChunkingSettings
from huflit_qa.processing.chunker import chunk_quy_che, chunk_thong_bao

CFG = ChunkingSettings(size=400, overlap=0, min_size=100)


def _record(content, doc_type="Quy chế"):
    return {
        "id": "42_1",
        "doc_title": "Quy chế đào tạo",
        "doc_type": doc_type,
        "doc_url": "https://portal.huflit.edu.vn/News/Detail/42/x",
        "date_posted": "2024-01-01",
        "content": content,
    }


def test_quy_che_splits_on_dieu_and_keeps_preamble():
    text = (
        "QUY CHẾ ĐÀO TẠO\n\n"
        "Điều 1. Phạm vi điều chỉnh\nQuy chế này áp dụng cho sinh viên đại học chính quy.\n\n"
        "Điều 2. Đối tượng áp dụng\nSinh viên, giảng viên và các đơn vị trong trường."
    )
    chunks = chunk_quy_che(_record(text), CFG)
    assert [c["chunk_id"] for c in chunks] == ["42_1_d0", "42_1_d1"]
    assert chunks[0]["text"].startswith("QUY CHẾ ĐÀO TẠO")
    assert chunks[1]["chunk_title"] == "Điều 2. Đối tượng áp dụng"
    assert chunks[0]["doc_type"] == "quy_che"


def test_long_dieu_is_split_at_khoan():
    khoan = "\n".join(f"{i}. " + "Nội dung khoản quy định chi tiết " * 5 for i in range(1, 5))
    chunks = chunk_quy_che(_record("Điều 5. Học phí\n" + khoan), CFG)
    assert len(chunks) > 1
    assert all(len(c["text"]) >= CFG.min_size for c in chunks)


def test_overlap_prepends_tail_of_previous_segment():
    paragraphs = "\n\n".join(f"Đoạn {i}: " + "x" * 150 for i in range(4))
    cfg = ChunkingSettings(size=400, overlap=20, min_size=100)
    chunks = chunk_thong_bao(_record(paragraphs, doc_type="Thông báo"), cfg)
    assert chunks[1]["text"].startswith(chunks[0]["text"][-20:])


def test_empty_content_yields_no_chunks():
    assert chunk_quy_che(_record("   "), CFG) == []
