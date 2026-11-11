"""Deploy the search-only demo to a Hugging Face Space (Docker SDK).

    python scripts/deploy_space.py                       # -> spaces/bebaobanhbaoo/huflit-regulation-search

The Space builds the repository's Dockerfile from a snapshot of the sources and the shipped index;
answer generation is switched off through a Space variable because free Spaces have no GPU.
Needs a Hugging Face token with write access (``huggingface-cli login``).
"""

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Dockerfile", ".dockerignore", "pyproject.toml"]
FOLDERS = ["src", "data/index"]


def stage(target: Path) -> None:
    for name in FILES:
        shutil.copy2(ROOT / name, target / name)
    for folder in FOLDERS:
        shutil.copytree(
            ROOT / folder,
            target / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "qdrant"),
        )
    # The Space card doubles as the README that pyproject.toml points to.
    shutil.copy2(ROOT / "deploy" / "space" / "README.md", target / "README.md")


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="bebaobanhbaoo/huflit-regulation-search")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.add_space_variable(args.repo, "HUFLIT_GENERATION__ENABLED", "false")
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(
            folder_path=tmp,
            repo_id=args.repo,
            repo_type="space",
            commit_message="Deploy search-only demo",
            delete_patterns=["src/**", "data/**"],
        )
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
