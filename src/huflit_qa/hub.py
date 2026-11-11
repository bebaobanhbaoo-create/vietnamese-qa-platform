"""Move the corpus to and from a Hugging Face Hub dataset repository.

The dataset holds what git should not: crawled pages, attachments (PDF + extracted text), processed
records and the shipped index. ``huflit-qa download-data`` restores them into ``data/``.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "bebaobanhbaoo/huflit-regulations"
DATA_FOLDERS = ("raw", "processed", "attachments", "index")


def upload_data(data_dir: Path, repo_id: str, private: bool = True, card: Path | None = None) -> str:
    """Create the dataset repo if needed and upload ``data/<folder>`` for each folder in DATA_FOLDERS."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    if card is not None:
        api.upload_file(
            path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset"
        )
    for folder in DATA_FOLDERS:
        logger.info("Uploading data/%s", folder)
        api.upload_folder(
            folder_path=str(data_dir / folder),
            path_in_repo=folder,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload data/{folder}",
        )
    return f"https://huggingface.co/datasets/{repo_id}"


def download_data(data_dir: Path, repo_id: str, folders: tuple[str, ...] = DATA_FOLDERS) -> Path:
    """Download the dataset into ``data_dir`` (private repos need ``huggingface-cli login`` or HF_TOKEN)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=[f"{f}/**" for f in folders],
    )
    return Path(path)
