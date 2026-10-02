# MỤC ĐÍCH: Phân loại bài viết thành doc_type (Thông báo / Quy chế)
#           và issuer_unit (đơn vị ban hành) dựa vào category_id.
#           Sau đó tách bài viết ra thành các file riêng theo doc_type.
#
# INPUT:  articles.json (list bài viết thô từ bước crawl)
# OUTPUT: data/processed/by_doc_type/thong_bao.json
#         data/processed/by_doc_type/quy_che.json
import re
import unicodedata
from pathlib import Path

from storage import load_json, save_json


CATEGORY_ID_MAPPING = {
    1:    {"doc_type": "Thông báo", "issuer_unit": "Chung"},
    11:   {"doc_type": "Thông báo", "issuer_unit": "Phòng Đào tạo"},
    13:   {"doc_type": "Thông báo", "issuer_unit": "Phòng CTSV"},
    14:   {"doc_type": "Thông báo", "issuer_unit": "Phòng Tài chính"},
    1011: {"doc_type": "Thông báo", "issuer_unit": "Phòng Khảo thí"},
    1023: {"doc_type": "Thông báo", "issuer_unit": "Phòng CTSV"},
    1025: {"doc_type": "Thông báo", "issuer_unit": "Phòng CTSV"},
    1026: {"doc_type": "Thông báo", "issuer_unit": "Bộ phận Y tế"},
    1016: {"doc_type": "Quy chế",   "issuer_unit": "Chung"},
}


def slugify(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_only).strip("_")
    return slug or "unknown"


def classify_articles(articles: list) -> list:
    classified = []
    for article in articles:
        category_id = article.get("category_id")
        if not category_id:
            continue
        mapping = CATEGORY_ID_MAPPING.get(int(category_id))
        if not mapping:
            continue
        enriched = dict(article)
        enriched["doc_type"] = mapping["doc_type"]
        enriched["issuer_unit"] = mapping["issuer_unit"]
        classified.append(enriched)
    return classified


def build_processed_outputs(raw_articles_path: Path, processed_dir: Path) -> list:
    classified = classify_articles(load_json(raw_articles_path))

    out_dir = processed_dir / "by_doc_type"
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    for article in classified:
        doc_type = article["doc_type"]
        if doc_type not in grouped:
            grouped[doc_type] = []
        grouped[doc_type].append(article)

    for doc_type, items in sorted(grouped.items()):
        save_json(items, out_dir / f"{slugify(doc_type)}.json")

    return classified
