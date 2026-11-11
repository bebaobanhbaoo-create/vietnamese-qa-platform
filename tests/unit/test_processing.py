import json

from huflit_qa.processing.attachments import find_attachment_file, read_txt_content, resolve_attachment_title
from huflit_qa.processing.flatten import flatten_all, is_placeholder
from huflit_qa.processing.records import build_flat_records
from huflit_qa.processing.sync import sync_records


def _doc(doc_id, content, attachments=()):
    return {
        "id": doc_id,
        "title": "Quy chế đào tạo",
        "url": f"https://portal.huflit.edu.vn/News/Detail/{doc_id}",
        "date_posted": "2024-11-25",
        "date_updated": None,
        "doc_type": "Quy chế",
        "category_name": "Quy dinh - Quy che",
        "issuer_unit": "Chung",
        "content": content,
        "attachments": list(attachments),
    }


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_flat_records_from_web_page_and_attachments(tmp_path):
    root = tmp_path
    att_dir = root / "data" / "attachments" / "quy_ch"
    folder = att_dir / "3208_Quy_che"
    folder.mkdir(parents=True)
    (folder / "3208_1__quy_che.txt").write_text("Điều 1. Phạm vi", encoding="utf-8")
    (folder / "3208_2__bang_diem.jpg").write_bytes(b"\xff\xd8\xff")

    source = root / "quy_che.json"
    attachments = [{"text": "Quy chế 2021", "url": "u1"}, {"text": "Bấm vào để xem", "url": "u2"}]
    _write(
        source,
        [_doc(3208, "Ban hành quy chế.\nBảng điểm\nBấm vào để xem", attachments), _doc(3300, "Nội dung")],
    )

    records = build_flat_records(source, att_dir, root)
    by_id = {r["id"]: r for r in records}
    assert set(by_id) == {"3208_1", "3208_2", "3300"}
    assert by_id["3208_1"]["content"] == "Điều 1. Phạm vi"
    assert by_id["3208_1"]["pdf_path"].startswith("data/attachments/quy_ch/3208_Quy_che/")
    assert by_id["3208_2"]["content_status"] == "image"
    assert by_id["3300"]["content_source"] == "web" and by_id["3300"]["content"] == "Nội dung"


def test_attachment_helpers(tmp_path):
    assert find_attachment_file(tmp_path, 1, 1) == (None, None)
    assert resolve_attachment_title("Quy chế 2021", 1, "") == "Quy chế 2021"
    content = "Mở đầu\nQuy định học bổng\nBấm vào để xem\nPhụ lục điểm\nBấm vào để xem"
    assert resolve_attachment_title("Bấm vào để xem", 2, content) == "Phụ lục điểm"
    assert resolve_attachment_title("Bấm vào để xem", 9, content) == "Tài liệu đính kèm 9"
    assert read_txt_content(tmp_path / "missing.txt") == (None, "empty")


def test_flatten_announcements_marks_placeholders(tmp_path):
    source = tmp_path / "thong_bao.json"
    docs = [
        {**_doc(1, "  Lịch thi học kỳ 1  "), "doc_type": "Thông báo"},
        {**_doc(2, "Vui lòng xem tại đây"), "doc_type": "Thông báo"},
    ]
    _write(source, docs)
    records = flatten_all(source)
    assert records[0]["content"] == "Lịch thi học kỳ 1" and records[0]["issuer_unit"] == "Chung"
    assert records[1]["content"] is None and records[1]["content_status"] == "empty"
    assert is_placeholder("  ") and not is_placeholder("Nội dung")


def test_sync_reloads_hand_corrected_text(tmp_path):
    txt = tmp_path / "data" / "a.txt"
    txt.parent.mkdir(parents=True)
    txt.write_text("Bản đã sửa tay", encoding="utf-8")
    flat = tmp_path / "flat.json"
    _write(
        flat,
        [
            {
                "id": "1_1",
                "pdf_path": "data/a.txt",
                "content_source": "pdf",
                "content_status": "empty",
                "content": None,
            }
        ],
    )
    assert sync_records(flat, tmp_path) == 1
    assert json.loads(flat.read_text(encoding="utf-8"))[0]["content"] == "Bản đã sửa tay"
    assert sync_records(flat, tmp_path) == 0
