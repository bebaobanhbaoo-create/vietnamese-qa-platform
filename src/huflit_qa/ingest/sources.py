"""Scraper configuration: portal URL, categories to crawl, request settings and link filters."""

from typing import Any

BASE_URL = "https://portal.huflit.edu.vn"

# Keep only categories for "Thong bao" and "Quy che"
CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "Thong bao chung"},
    {"id": 11, "name": "Thong bao Phong Dao tao Dai hoc"},
    {"id": 13, "name": "Thong bao Phong Chinh tri - Cong tac Sinh vien"},
    {"id": 14, "name": "Thong bao Phong Ke hoach - Tai chinh"},
    {"id": 1011, "name": "Thong bao Phong Khao thi - Bao dam chat luong giao duc"},
    {"id": 1023, "name": "Hoc bong - Khen thuong"},
    {"id": 1025, "name": "Thong bao Tuan sinh hoat cong dan - Sinh vien"},
    {"id": 1026, "name": "Thong bao Bo phan Y te"},
    {"id": 1016, "name": "Quy dinh - Quy che"},
]

# Request settings
REQUEST_DELAY = 0.5
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Domains to ignore from page navigation noise
EXCLUDED_DOMAINS = {
    "portal.huflit.edu.vn",
    "huflit.edu.vn",
    "lib.huflit.edu.vn",
    "pga.huflit.edu.vn",
    "font",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
}

FILE_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".rar",
    ".7z",
    ".txt",
    ".csv",
    ".odt",
    ".ods",
}

ATTACHMENT_DOMAINS = {
    "sharepoint.com": "sharepoint",
    "onedrive.live.com": "onedrive",
    "drive.google.com": "google_drive",
    "docs.google.com": "google_docs",
    "forms.office.com": "ms_form",
    "forms.gle": "google_form",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "fb.com": "facebook",
    "facebook.com": "facebook",
    "zalo.me": "zalo",
}
