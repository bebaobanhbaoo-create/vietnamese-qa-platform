from huflit_qa.evaluation.metrics import ndcg_from_hits, recall_from_hits
from huflit_qa.evaluation.projection import build_projection, project_ranking
from huflit_qa.evaluation.retrieval_eval import score_run
from huflit_qa.indexing.tokenize import pyvi_tokens, whitespace_tokens, word_tokens

A = "sinh viên được xét tốt nghiệp khi tích lũy đủ số tín chỉ của chương trình đào tạo"
B = "sinh viên phải đạt chuẩn đầu ra ngoại ngữ và tin học trước khi xét tốt nghiệp"
UNITS = [
    {"unit_id": "u1", "doc_id": "d", "text": A},
    {"unit_id": "u2", "doc_id": "d", "text": B},
    {"unit_id": "u3", "doc_id": "other", "text": A},
]


def test_identical_chunk_maps_to_itself_only():
    assert build_projection([{"chunk_id": "u1", "doc_id": "d", "text": A}], UNITS) == {"u1": ["u1"]}


def test_merged_chunk_covers_both_units_of_same_document():
    proj = build_projection([{"chunk_id": "c0", "doc_id": "d", "text": A + ". " + B}], UNITS)
    assert proj == {"c0": ["u1", "u2"]}  # not u3: different source record


def test_piece_of_a_unit_maps_to_it():
    piece = " ".join(A.split()[:8])
    assert build_projection([{"chunk_id": "c0", "doc_id": "d", "text": piece}], UNITS) == {"c0": ["u1"]}


def test_project_ranking_dedupes_in_order():
    assert project_ranking(["c0", "c1"], {"c0": ["u2", "u1"], "c1": ["u1", "u3"]}) == ["u2", "u1", "u3"]


def test_score_run_identity_projection_matches_plain_scoring():
    queries = [{"relevant_chunk_ids": ["u1", "u2"]}]
    plain = score_run([["x", "u1", "u2"]], queries, ks=(3,))
    projected = score_run(
        [["x", "u1", "u2"]], queries, ks=(3,), projection={"x": [], "u1": ["u1"], "u2": ["u2"]}
    )
    assert plain == projected


def test_chunk_covering_two_relevant_units_counts_once_for_ndcg():
    hits = [{"u1", "u2"}]
    assert recall_from_hits(hits, 2, 1) == 1.0
    assert ndcg_from_hits(hits, 2, 10) < 1.0


def test_tokenizers():
    text = "Điều kiện xét tốt nghiệp."
    assert whitespace_tokens(text)[-1] == "nghiệp."
    assert word_tokens(text)[-1] == "nghiệp"
    assert "tốt_nghiệp" in pyvi_tokens(text)


def test_paired_bootstrap_detects_consistent_improvement():
    from huflit_qa.evaluation.retrieval_eval import paired_bootstrap

    a = [0.2, 0.4, 0.5, 0.1, 0.3] * 6
    better = paired_bootstrap(a, [x + 0.1 for x in a])
    assert abs(better["diff"] - 0.1) < 1e-9
    assert better["ci_low"] > 0 and better["p_not_better"] == 0.0
    same = paired_bootstrap(a, a)
    assert same["diff"] == 0.0 and same["p_not_better"] == 1.0
