"""Command-line entry point: ``huflit-qa <command>``."""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from huflit_qa.config import get_settings
from huflit_qa.log import setup_logging

app = typer.Typer(
    help="HUFLIT regulation QA: scrape, process, index, query and evaluate.",
    no_args_is_help=True,
)
logger = logging.getLogger("huflit_qa")


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging")] = False) -> None:
    setup_logging(logging.DEBUG if verbose else logging.INFO)


@app.command()
def scrape(
    categories: Annotated[list[int] | None, typer.Option("--category", "-c", help="Category ID(s)")] = None,
    no_resume: Annotated[bool, typer.Option(help="Start from scratch")] = False,
    list_only: Annotated[bool, typer.Option(help="Only crawl list pages (metadata)")] = False,
) -> None:
    """Step 1: crawl the student portal into data/raw and data/processed/by_doc_type."""
    from huflit_qa.ingest.scrape import scrape as run_scrape

    run_scrape(get_settings(), categories, resume=not no_resume, list_only=list_only)


@app.command("normalize-attachments")
def normalize_attachments(
    delete_html: Annotated[bool, typer.Option(help="Delete .bin files that are HTML error pages")] = False,
) -> None:
    """Rename downloaded .bin attachments to their real file type."""
    from huflit_qa.ingest.attachments import normalize_attachments as run

    stats = run(get_settings().attachments_dir, delete_html)
    logger.info("converted=%d deleted_html=%d unknown=%d", stats.converted, stats.deleted_html, stats.unknown)


@app.command("extract-pdf")
def extract_pdf(
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing (possibly hand-edited) .txt")] = False,
) -> None:
    """Extract text from PDF attachments into .txt files next to them."""
    from huflit_qa.ingest.pdf import extract_missing_txt

    written = extract_missing_txt(get_settings().attachments_dir, overwrite)
    logger.info("Wrote %d .txt files", written)


@app.command()
def process() -> None:
    """Step 2: build the flat quy_che / thong_bao records."""
    from huflit_qa.pipeline import process as run

    run(get_settings())


@app.command()
def sync() -> None:
    """Reload hand-corrected attachment .txt files into quy_che_flat.json."""
    from huflit_qa.processing.sync import sync_records

    settings = get_settings()
    updated = sync_records(settings.quy_che_flat_path, settings.root_dir)
    logger.info("Sync complete: %d updated", updated)


@app.command()
def index() -> None:
    """Step 3: chunk records and build the BM25 + FAISS indexes."""
    from huflit_qa.pipeline import build_index

    build_index(get_settings())


@app.command()
def query(
    text: Annotated[str, typer.Argument(help="Question in Vietnamese")],
    top_k: Annotated[int | None, typer.Option("--top-k", "-k")] = None,
) -> None:
    """Step 4: hybrid search for a question."""
    from huflit_qa.retrieval.hybrid import HybridRetriever

    results = HybridRetriever(get_settings()).retrieve(text, top_k=top_k)
    typer.echo(f"\nQuery: {text}\nTop {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        typer.echo(f"[{i}] {r['chunk_title'] or r['doc_title']}")
        typer.echo(f"    Score: {r['score']:.4f}")
        typer.echo(f"    URL: {r['source_url']}")
        typer.echo(f"    {r['text'][:200]}...\n")


@app.command("eval")
def evaluate(
    mode: Annotated[str, typer.Option(help="bm25 | dense | hybrid | all")] = "all",
    split: Annotated[str, typer.Option(help="dev (for tuning) | test (for reporting) | all")] = "all",
    output: Annotated[Path | None, typer.Option(help="Save the report as JSON")] = None,
) -> None:
    """Step 5: evaluate retrieval on eval/testset.jsonl."""
    from huflit_qa.evaluation.retrieval_eval import MODES, format_report, run_evaluation

    if mode != "all" and mode not in MODES:
        raise typer.BadParameter(f"mode must be one of {(*MODES, 'all')}")
    if split not in ("dev", "test", "all"):
        raise typer.BadParameter("split must be dev, test or all")
    modes = MODES if mode == "all" else (mode,)

    report = run_evaluation(get_settings(), modes, split=None if split == "all" else split)
    typer.echo(format_report(report))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"\nSaved -> {output}")


@app.command()
def pool(
    output: Annotated[Path, typer.Option(help="Pool file for the annotation page")] = Path(
        "eval/annotation/pool.json"
    ),
    depth: Annotated[int, typer.Option(help="Top-N taken from each retriever")] = 10,
    only_unlabelled: Annotated[bool, typer.Option(help="Skip queries already labelled")] = False,
    indexes: Annotated[
        list[Path] | None,
        typer.Option("--index", help="Index directories to pool from (repeatable; default: data/index)"),
    ] = None,
) -> None:
    """Pool label units retrieved by BM25, dense and hybrid search of one or more indexes."""
    from huflit_qa.evaluation.pooling import build_pool, precomputed_searcher, save_pool, unit_rankings
    from huflit_qa.evaluation.projection import load_units
    from huflit_qa.evaluation.testset import load_testset
    from huflit_qa.retrieval.hybrid import HybridRetriever

    settings = get_settings()
    queries = load_testset(settings.testset_path)
    if only_unlabelled:
        queries = [q for q in queries if not q.get("labelled")]
    units = load_units(settings.label_units_path)

    # Run every index over all queries first, one model in memory at a time.
    rankings: dict[str, dict[str, list[str]]] = {}
    for index_dir in indexes or [settings.index_dir]:
        retriever = HybridRetriever(settings.model_copy(update={"index_path": index_dir}))
        for mode, by_query in unit_rankings(retriever, queries, units, depth).items():
            rankings[f"{index_dir.name}/{mode}"] = by_query
        del retriever

    searchers = {name: precomputed_searcher(by_query) for name, by_query in rankings.items()}
    unit_map = {u["unit_id"]: u for u in units}
    result = build_pool(queries, searchers, unit_map, depth)
    save_pool(result, output)
    sizes = [len(p["candidates"]) for p in result]
    logger.info(
        "%d queries, %d candidates (avg %.1f) -> %s",
        len(result),
        sum(sizes),
        sum(sizes) / max(len(sizes), 1),
        output,
    )


@app.command("merge-labels")
def merge_labels(
    labels_file: Annotated[Path, typer.Argument(help="labels.json exported by the annotation page")],
) -> None:
    """Write labels exported from eval/annotation/annotate.html into eval/testset.jsonl."""
    from huflit_qa.evaluation.pooling import merge_labels as merge
    from huflit_qa.evaluation.testset import load_testset, save_testset

    settings = get_settings()
    queries = load_testset(settings.testset_path)
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    updated = merge(queries, labels)
    save_testset(queries, settings.testset_path)
    logger.info("Updated %d queries in %s", updated, settings.testset_path)


if __name__ == "__main__":
    app()
