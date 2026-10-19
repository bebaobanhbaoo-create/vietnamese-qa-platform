# MỤC ĐÍCH: Gửi HTTP request đến website HUFLIT và lấy nội dung HTML.
#           Hỗ trợ tự động retry khi gặp lỗi mạng, giới hạn tốc độ request.
#
# HTTP request — yêu cầu gửi đến máy chủ web để lấy nội dung trang.
#   Giống như bạn gõ địa chỉ web vào trình duyệt, nhưng làm bằng code Python.
#
# INPUT:  URL (địa chỉ trang web)
# OUTPUT: Chuỗi HTML của trang đó (hoặc None nếu thất bại)
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from huflit_qa.ingest.parser import get_total_pages, parse_detail_page, parse_list_page
from huflit_qa.ingest.sources import (
    BASE_URL,
    CATEGORIES,
    HEADERS,
    MAX_RETRIES,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
)


class RateLimitedSession:
    """HTTP session với retry tự động và giới hạn tốc độ request."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))
        self._session.mount("http://", HTTPAdapter(max_retries=retry))
        self._last_request_time = 0.0

    def get(self, url: str):
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        try:
            resp = self._session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            self._last_request_time = time.time()
            return resp
        except requests.exceptions.RequestException:
            return None


class HuflitCrawler:
    def __init__(self):
        self._session = RateLimitedSession()

    def _fetch(self, url: str):
        resp = self._session.get(url)
        if resp is None:
            return None
        try:
            return resp.content.decode("utf-8")
        except UnicodeDecodeError:
            return resp.text

    def crawl_category(self, cat_id: int, cat_name: str) -> list:
        """Crawl tất cả pages của một category, trả về list article metadata."""
        html = self._fetch(f"{BASE_URL}/News/Type/{cat_id}")
        if not html:
            return []

        total_pages = get_total_pages(html)
        all_articles = parse_list_page(html, cat_id, cat_name)

        for page in range(2, total_pages + 1):
            html = self._fetch(f"{BASE_URL}/News/Type/{cat_id}?page={page}")
            if not html:
                continue
            all_articles.extend(parse_list_page(html, cat_id, cat_name))

        seen = set()
        deduped = []
        for art in all_articles:
            if art["id"] not in seen:
                seen.add(art["id"])
                deduped.append(art)
        return deduped

    def crawl_all_categories(self) -> list:
        """Crawl tất cả categories."""
        all_metas = []
        for cat in CATEGORIES:
            all_metas.extend(self.crawl_category(cat["id"], cat["name"]))
        return all_metas

    def crawl_detail(self, article_meta: dict):
        """Crawl và parse trang chi tiết một bài viết."""
        html = self._fetch(article_meta["url"])
        if not html:
            return None
        return parse_detail_page(html, article_meta)

    def crawl_all_details(self, article_metas: list) -> list:
        """Crawl toàn bộ detail pages."""
        results = []
        for meta in article_metas:
            detail = self.crawl_detail(meta)
            if detail:
                results.append(detail)
            else:
                fallback = dict(meta)
                fallback["content"] = ""
                fallback["attachments"] = []
                fallback["attachment_count"] = 0
                fallback["fetch_error"] = True
                results.append(fallback)
        return results
