import pytest

from huflit_qa.evaluation.pooling import build_pool, merge_labels
from huflit_qa.evaluation.testset import assign_split, load_testset, save_testset, select


def _q(qid, relevant, labelled=True, split="dev"):
    return {
        "id": qid,
        "query": f"câu hỏi {qid}",
        "relevant_chunk_ids": relevant,
        "labelled": labelled,
        "split": split,
    }


def test_split_is_deterministic_and_roughly_balanced():
    splits = [assign_split(f"q{i:03d}") for i in range(1, 1001)]
    assert splits == [assign_split(f"q{i:03d}") for i in range(1, 1001)]
    assert 0.3 < splits.count("dev") / len(splits) < 0.5


def test_select_skips_unlabelled_and_no_answer_queries():
    queries = [
        _q("a", ["c1"]),
        _q("b", [], labelled=True),
        _q("c", [], labelled=False),
        _q("d", ["c2"], split="test"),
    ]
    assert [q["id"] for q in select(queries)] == ["a", "d"]
    assert [q["id"] for q in select(queries, "test")] == ["d"]


def test_jsonl_round_trip(tmp_path):
    queries = [_q("a", ["c1"]), _q("b", [])]
    save_testset(queries, tmp_path / "t.jsonl")
    assert load_testset(tmp_path / "t.jsonl") == queries


CHUNKS = {cid: {"doc_title": "T", "chunk_title": None, "source_url": "u", "text": cid} for cid in "abcdz"}


def test_pool_unions_systems_and_keeps_existing_labels():
    searchers = {"s1": lambda q, n: ["a", "b"][:n], "s2": lambda q, n: ["b", "c"][:n]}
    [entry] = build_pool([_q("q1", ["z"])], searchers, CHUNKS, depth=2)
    ids = [c["chunk_id"] for c in entry["candidates"]]
    assert set(ids) == {"a", "b", "c", "z"}
    assert ids[-1] == "z"  # found only through the old label
    b = next(c for c in entry["candidates"] if c["chunk_id"] == "b")
    assert b["ranks"] == {"s1": 2, "s2": 1}
    assert next(c for c in entry["candidates"] if c["chunk_id"] == "z")["labelled_relevant"]


def test_merge_labels_overwrites_and_marks_labelled():
    queries = [_q("q1", ["old"], labelled=False)]
    assert merge_labels(queries, [{"id": "q1", "relevant_chunk_ids": ["b", "a", "a"]}]) == 1
    assert queries[0]["relevant_chunk_ids"] == ["a", "b"]
    assert queries[0]["labelled"] is True
    with pytest.raises(KeyError):
        merge_labels(queries, [{"id": "nope", "relevant_chunk_ids": []}])
