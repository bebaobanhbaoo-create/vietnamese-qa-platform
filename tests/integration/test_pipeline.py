"""End to end on a tiny corpus: flat records -> chunks -> BM25 + FAISS -> hybrid search -> API.

Uses the real embedding model (MiniLM, ~0.5 GB download, cached in CI), so it is skipped when
sentence-transformers is not installed.
"""

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")

from huflit_qa.config import Settings
from huflit_qa.generation.llm import FakeLLM
from huflit_qa.generation.rag import RAGPipeline
from huflit_qa.pipeline import build_index
from huflit_qa.retrieval.filters import SearchFilters
from huflit_qa.retrieval.hybrid import HybridRetriever

REGULATION = """QUY CHẾ ĐÀO TẠO TRÌNH ĐỘ ĐẠI HỌC
Ban hành kèm theo Quyết định ngày 30 tháng 12 năm 2021

Điều 9. Tổ chức đăng ký học tập
Số tín chỉ đăng ký trong mỗi học kỳ không vượt quá 3/2 khối lượng trung bình một học kỳ
theo kế hoạch học tập chuẩn.

Điều 12. Đăng ký học lại
Sinh viên có học phần bắt buộc bị điểm F phải đăng ký học lại học phần đó ở một trong các học kỳ tiếp theo.

Điều 20. Điều kiện xét công nhận tốt nghiệp
Sinh viên được xét công nhận tốt nghiệp khi tích lũy đủ số tín chỉ, đạt chuẩn đầu ra ngoại ngữ và tin học."""

OLD_REGULATION = """QUY CHẾ ĐÀO TẠO (CŨ)
Ban hành ngày 15 tháng 8 năm 2007

Điều 9. Khối lượng học tập
Khối lượng học tập tối thiểu mà mỗi sinh viên phải đăng ký trong mỗi học kỳ là 14 tín chỉ."""

ANNOUNCEMENT = (
    "Nhà trường thông báo lịch thi kết thúc học phần học kỳ 1 năm học 2025-2026 bắt đầu từ ngày 5/1/2026."
)


def _quy_che(record_id, content):
    return {
        "id": record_id,
        "doc_title": "Quy chế đào tạo",
        "doc_url": f"https://portal.huflit.edu.vn/News/Detail/{record_id}",
        "date_posted": "2024-11-25",
        "doc_type": "Quy chế",
        "issuer_unit": "Chung",
        "content_status": "ok",
        "content": content,
    }


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("corpus")
    by_type = root / "data" / "processed" / "by_doc_type"
    by_type.mkdir(parents=True)
    quy_che = [_quy_che("100_1", OLD_REGULATION), _quy_che("100_2", REGULATION)]
    thong_bao = [
        {
            "id": "200",
            "doc_title": "Thông báo lịch thi kết thúc học phần học kỳ 1 năm học 2025-2026",
            "doc_url": "https://portal.huflit.edu.vn/News/Detail/200",
            "date_posted": "2025-12-01",
            "doc_type": "Thông báo",
            "issuer_unit": "Phòng Khảo thí",
            "content_status": "ok",
            "content": ANNOUNCEMENT,
        }
    ]
    (by_type / "quy_che_flat.json").write_text(json.dumps(quy_che, ensure_ascii=False), encoding="utf-8")
    (by_type / "thong_bao_flat.json").write_text(json.dumps(thong_bao, ensure_ascii=False), encoding="utf-8")
    settings = Settings(root_dir=Path(root))
    build_index(settings)
    return settings


@pytest.fixture(scope="module")
def retriever(settings) -> HybridRetriever:
    return HybridRetriever(settings)


def test_index_files_and_manifest(settings):
    for path in (settings.chunks_path, settings.bm25_path, settings.faiss_path, settings.manifest_path):
        assert path.exists()
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert manifest["num_chunks"] == len(json.loads(settings.chunks_path.read_text(encoding="utf-8")))
    assert manifest["embedding"]["dim"] == 384


@pytest.mark.parametrize(
    ("query", "expected_title"),
    [
        ("điều kiện xét tốt nghiệp", "Điều 20. Điều kiện xét công nhận tốt nghiệp"),
        ("bị điểm F thì phải làm gì", "Điều 12. Đăng ký học lại"),
    ],
)
def test_hybrid_search_finds_the_article(retriever, query, expected_title):
    top = retriever.retrieve(query, top_k=1)[0]
    assert top["chunk_title"] == expected_title


def test_filters_drop_superseded_regulation(retriever):
    query = "đăng ký tối đa bao nhiêu tín chỉ mỗi học kỳ"
    ids = {r["chunk_id"] for r in retriever.retrieve(query, top_k=5, filters=SearchFilters(newest_only=True))}
    assert ids and not any(cid.startswith("100_1") for cid in ids)
    only_announcements = retriever.retrieve("lịch thi", top_k=5, filters=SearchFilters(doc_type="thong_bao"))
    assert [r["chunk_id"] for r in only_announcements] == ["200_p0"]


def test_api_end_to_end(retriever):
    from fastapi.testclient import TestClient

    from huflit_qa.api.app import Services, create_app
    from huflit_qa.config import GenerationSettings

    llm = FakeLLM(answer="Được xét tốt nghiệp khi tích lũy đủ tín chỉ [1].")
    services = Services(retriever, RAGPipeline(retriever, llm, GenerationSettings()), llm, threading.Lock())
    with TestClient(create_app(services=services)) as client:
        body = client.post("/v1/ask", json={"question": "Điều kiện xét tốt nghiệp?"}).json()
    assert body["citations"]["grounded"]
    assert body["sources"][0]["chunk_title"] == "Điều 20. Điều kiện xét công nhận tốt nghiệp"
