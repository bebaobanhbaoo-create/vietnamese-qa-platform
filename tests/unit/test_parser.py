"""Parser tests on small pages shaped like portal.huflit.edu.vn, so a layout change fails CI."""

import pytest

pytest.importorskip("bs4")

from huflit_qa.ingest.parser import get_total_pages, parse_detail_page, parse_list_page

LIST_PAGE = """
<html><body><div class="divmain">
  <div class="item">
    <a href="/News/Detail/3208/Quy-che-dao-tao">Quy chế đào tạo</a>
    <i>Ngày đăng: 25/11/2024</i> <i>Ngày cập nhật: 01/12/2024</i>
  </div>
  <div class="item">
    <a href="/News/Detail/3612/Quy-dinh-quy-ho-tro">Quy định về việc sử dụng Quỹ hỗ trợ sinh viên</a>
    <i>Ngày đăng: 20/02/2025</i>
  </div>
  <a href="/News/Type/1016?page=2">2</a> <a href="/News/Type/1016?page=7">Trang cuối</a>
</div></body></html>
"""

DETAIL_PAGE = """
<html><body><div class="divmain">
  <div class="bgtitle">Quy định - Quy chế</div>
  <div><a class="title_topicdisplay">Quy chế đào tạo</a></div>
  <p>Nhà trường ban hành Quy chế đào tạo trình độ đại học.</p>
  <ul><li>Áp dụng từ khóa 2021</li><li>Thay thế quy chế cũ</li></ul>
  <p>Quy chế chi tiết <a href="https://drive.google.com/file/d/abc/view">tại đây</a></p>
  <p><a href="/files/quy-che.pdf">Quy chế đào tạo 2021.pdf</a></p>
  <p><a href="https://portal.huflit.edu.vn/News/Type/1016">Quay lại</a></p>
</div></body></html>
"""

META = {
    "id": 3208,
    "category_id": 1016,
    "category_name": "Quy dinh - Quy che",
    "title": "Quy chế đào tạo",
    "url": "https://portal.huflit.edu.vn/News/Detail/3208/Quy-che-dao-tao",
    "date_posted": "2024-11-25",
    "date_updated": None,
}


def test_list_page_articles_dates_and_pagination():
    articles = parse_list_page(LIST_PAGE, 1016, "Quy dinh - Quy che")
    assert [a["id"] for a in articles] == [3208, 3612]
    assert articles[0]["date_posted"] == "2024-11-25" and articles[0]["date_updated"] == "2024-12-01"
    assert articles[0]["url"] == "https://portal.huflit.edu.vn/News/Detail/3208/Quy-che-dao-tao"
    assert get_total_pages(LIST_PAGE) == 7


def test_detail_page_content_and_attachments():
    detail = parse_detail_page(DETAIL_PAGE, META)
    content = detail["content"]
    assert content.startswith("Nhà trường ban hành Quy chế đào tạo")
    assert "Áp dụng từ khóa 2021\nThay thế quy chế cũ" in content
    assert "Quy định - Quy chế" not in content  # breadcrumb removed

    by_type = {a["type"]: a for a in detail["attachments"]}
    assert set(by_type) == {"google_drive", "file_pdf"}
    # A vague link text ("tại đây") gets the surrounding text as context, minus vague phrases
    # such as "chi tiết tại đây".
    assert by_type["google_drive"]["context"] == "Quy chế"
    assert by_type["file_pdf"]["url"] == "https://portal.huflit.edu.vn/files/quy-che.pdf"
    assert detail["attachment_count"] == 2


def test_portal_hosted_files_are_attachments_but_navigation_is_not():
    from huflit_qa.ingest.parser import _is_boilerplate

    assert not _is_boilerplate("https://portal.huflit.edu.vn/Content/uploads/quy-che.pdf")
    assert not _is_boilerplate("https://portal.huflit.edu.vn/files/danh-sach.xlsx")
    assert _is_boilerplate("https://portal.huflit.edu.vn/")
    assert _is_boilerplate("https://portal.huflit.edu.vn/News/Type/1016")
    assert _is_boilerplate("https://portal.huflit.edu.vn/Login")
    assert not _is_boilerplate("https://portal.huflit.edu.vn/News/Detail/3208/x")
