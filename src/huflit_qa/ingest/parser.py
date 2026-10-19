"""Parse trang list và trang chi tiết bài viết. Extract attachments."""

import copy
import re
import unicodedata
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from huflit_qa.ingest.sources import ATTACHMENT_DOMAINS, BASE_URL, EXCLUDED_DOMAINS, FILE_EXTENSIONS

DATE_PATTERN = re.compile(r"(\d{2}/\d{2}/\d{4})")
ARTICLE_ID_PATTERN = re.compile(r"/News/Detail/(\d+)/", re.IGNORECASE)

BLOCK_TAGS = {
    "p",
    "div",
    "li",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
    "hr",
    "td",
    "th",
    "ul",
    "ol",
    "table",
    "blockquote",
    "section",
}

VAGUE_LINK_TEXTS = {
    "tại đây",
    "tai day",
    "xem tại đây",
    "tải tại đây",
    "click here",
    "please click here",
    "bấm vào để xem",
    "xem pdf tại đây",
    "đính kèm",
    "xem đính kèm",
    "danh sách đính kèm",
    "xem chi tiết",
    "chi tiết tại đây",
    "truy cập tại đây",
    "here",
    "link",
    "tại link",
    "download",
    "tải về",
    "tại đây >",
    "<<vui lòng xem tại đây>>",
    "<<please click here>>",
    "<<tải file tại đây>>",
    "(xem pdf tại đây)",
    "theo danh sách đính kèm",
    "<<vui lòng xem danh sách đính kèm tại đây>>",
    "xem hướng dẫn",
    "danh sách sinh viên",
    "tại đây.",
    "vui lòng xem tại đây",
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _parse_date(text: str):
    m = DATE_PATTERN.search(text)
    if not m:
        return None
    d, mo, y = m.group(1).split("/")
    return f"{y}-{mo}-{d}"


def parse_list_page(html: str, category_id: int, category_name: str) -> list:
    soup = BeautifulSoup(html, "lxml")
    articles: list[dict] = []

    divmain = soup.find("div", class_="divmain")
    if not divmain:
        return articles

    for link_tag in divmain.find_all("a", href=ARTICLE_ID_PATTERN):
        href = link_tag["href"].strip()
        id_match = ARTICLE_ID_PATTERN.search(href)
        if id_match is None:
            continue
        article_id = int(id_match.group(1))

        raw_title = link_tag.get_text(separator=" ", strip=True)
        title = re.sub(r"^[\s\u0000-\u001f\ue000-\uf8ff]+", "", raw_title).strip() or raw_title

        date_posted = None
        date_updated = None
        container = link_tag.parent
        for _ in range(4):
            if container is None or container == divmain:
                break
            for i_tag in container.find_all("i"):
                text = _normalize(i_tag.get_text(strip=True)).lower()
                if "ngày đăng" in text and not date_posted:
                    date_posted = _parse_date(text)
                elif "ngày cập nhật" in text and not date_updated:
                    date_updated = _parse_date(text)
            if date_posted or date_updated:
                break
            container = container.parent

        articles.append(
            {
                "id": article_id,
                "category_id": category_id,
                "category_name": category_name,
                "title": title,
                "url": BASE_URL + href,
                "date_posted": date_posted,
                "date_updated": date_updated,
            }
        )

    return articles


def get_total_pages(html: str) -> int:
    soup = BeautifulSoup(html, "lxml")
    max_page = 1
    for tag in soup.find_all("a", href=True):
        text = tag.get_text(strip=True).lower()
        if "trang cuối" in text or "last" in text:
            m = re.search(r"[?&]page=(\d+)", tag["href"])
            if m:
                return int(m.group(1))
        m = re.search(r"[?&]page=(\d+)", tag["href"])
        if m:
            p = int(m.group(1))
            if p > max_page:
                max_page = p
    return max_page


def _extract_content_text(element) -> str:
    parts = []

    def walk(node):
        if isinstance(node, NavigableString):
            text = str(node)
            text = re.sub(r"[\r\n\t]+", " ", text)
            text = re.sub(r" {2,}", " ", text)
            if text.strip():
                parts.append(text)
        elif isinstance(node, Tag):
            name = node.name.lower() if node.name else ""
            if name in ("script", "style"):
                return
            is_block = name in BLOCK_TAGS
            if is_block and parts and parts[-1] not in ("\n", "\n\n") and parts[-1].rstrip():
                parts.append("\n")
            for child in node.children:
                walk(child)
            if is_block and parts and parts[-1] not in ("\n", "\n\n"):
                parts.append("\n")

    walk(element)

    text = "".join(parts)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +([,\.;:!?])", r"\1", text)
    return _normalize(text).strip()


def parse_detail_page(html: str, article_meta: dict) -> dict:
    soup = BeautifulSoup(html, "lxml")
    page_url = article_meta["url"]
    divmain = soup.find("div", class_="divmain")

    title = article_meta["title"]
    content_text = ""

    if divmain:
        title_tag = divmain.find("a", class_="title_topicdisplay")
        if title_tag:
            t = title_tag.get_text(separator=" ", strip=True)
            if t:
                title = t

        for unwanted in divmain.find_all("div", class_="bgtitle"):
            unwanted.decompose()
        title_div = divmain.find("a", class_="title_topicdisplay")
        if title_div and title_div.parent:
            title_div.parent.decompose()
        for unwanted in divmain.find_all(["script", "style"]):
            unwanted.decompose()
        content_text = _extract_content_text(divmain)

    date_posted = article_meta.get("date_posted")
    date_updated = article_meta.get("date_updated")

    if not date_posted or not date_updated:
        html_nfc = _normalize(html)
        if not date_posted:
            m = re.search(r"ngày đăng\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})", html_nfc, re.IGNORECASE)
            if m:
                date_posted = _parse_date(m.group(1))
        if not date_updated:
            m = re.search(r"ngày cập nhật\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})", html_nfc, re.IGNORECASE)
            if m:
                date_updated = _parse_date(m.group(1))

    fresh_soup = BeautifulSoup(html, "lxml")
    attachments = extract_attachments(fresh_soup, page_url)

    return {
        "id": article_meta["id"],
        "category_id": article_meta["category_id"],
        "category_name": article_meta["category_name"],
        "title": title,
        "url": page_url,
        "date_posted": date_posted,
        "date_updated": date_updated,
        "content": content_text,
        "attachments": attachments,
        "attachment_count": len(attachments),
    }


# --- Attachment extraction ---


def _clean_text(text: str) -> str:
    text = _normalize(text)
    text = re.sub(r"[\s\xa0]+", " ", text).strip()
    text = re.sub(r"^[<{(\[]+|[>})\]]+$", "", text).strip()
    return text


def _is_vague(text: str) -> bool:
    cleaned = _clean_text(text).lower()
    return cleaned in VAGUE_LINK_TEXTS or len(cleaned) < 4


def _get_context_text(tag) -> str:
    parents_to_check = [tag.parent]
    if tag.parent:
        parents_to_check.append(tag.parent.parent)
    for parent in parents_to_check:
        if parent is None:
            break
        if parent.name in ("p", "li", "td", "span", "div", "strong", "em"):
            ctx = _clean_text(parent.get_text(separator=" ", strip=True))
            for vague in VAGUE_LINK_TEXTS:
                ctx = re.sub(re.escape(vague) + r"\s*$", "", ctx, flags=re.IGNORECASE).strip()
            ctx = re.sub(r"[<{(\[]+\s*$", "", ctx).strip()
            ctx = re.sub(r"[,;:\.]+$", "", ctx).strip()
            if ctx and len(ctx) > 5:
                return ctx
    return ""


def _classify_url(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    for domain, atype in ATTACHMENT_DOMAINS.items():
        if domain in netloc:
            return atype
    path = parsed.path.lower()
    for ext in FILE_EXTENSIONS:
        if path.endswith(ext):
            return f"file_{ext.lstrip('.')}"
    if netloc and netloc not in EXCLUDED_DOMAINS:
        return "external_link"
    return "internal"


def _is_boilerplate(url: str) -> bool:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    if not netloc or netloc == "portal.huflit.edu.vn":
        boilerplate_paths = ["/", "/login", "/news/type/", "/public/tracuu", "/content/"]
        for bp in boilerplate_paths:
            if path.startswith(bp):
                return True
        return not path.startswith("/news/detail/")
    return any(netloc == excluded or netloc.endswith("." + excluded) for excluded in EXCLUDED_DOMAINS)


def extract_attachments(soup, page_url: str) -> list:
    attachments: list[dict] = []
    seen_urls = set()

    content_div = (
        soup.find("div", class_="divmain")
        or soup.find("div", id="divNews")
        or soup.find("div", class_="content")
        or soup.find("article")
        or soup.body
    )
    if not content_div:
        return attachments

    content_div = copy.copy(content_div)
    left_menu_divs = []
    for div in content_div.find_all("div"):
        div_id = div.get("id", "")
        if div_id and "leftMenu" in div_id:
            left_menu_divs.append(div)
    for unwanted in left_menu_divs:
        unwanted.decompose()
    for unwanted in content_div.find_all("div", class_="bgtitle"):
        unwanted.decompose()
    for unwanted in content_div.find_all("a", class_="title_topicdisplay"):
        unwanted.decompose()
    for a in content_div.find_all("a", href=re.compile(r"[?&]page=\d+")):
        a.decompose()

    for tag in content_div.find_all("a", href=True):
        raw_href = tag["href"].strip()
        if not raw_href or raw_href.startswith("#") or raw_href.startswith("javascript:"):
            continue

        if raw_href.startswith("//"):
            full_url = "https:" + raw_href
        elif raw_href.startswith("/"):
            full_url = BASE_URL + raw_href
        elif not raw_href.startswith("http"):
            full_url = urljoin(page_url, raw_href)
        else:
            full_url = raw_href

        if _is_boilerplate(full_url) or full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        link_type = _classify_url(full_url)
        if link_type == "internal":
            continue

        link_text = _clean_text(tag.get_text(separator=" ", strip=True))
        if not link_text:
            link_text = _clean_text(tag.get("title", "") or tag.get("alt", "") or "")

        context = _get_context_text(tag) if _is_vague(link_text) else ""

        text_value = link_text or full_url
        attachments.append(
            {
                "text": text_value,
                "context": context,
                "url": full_url,
                "type": link_type,
            }
        )

    return attachments
