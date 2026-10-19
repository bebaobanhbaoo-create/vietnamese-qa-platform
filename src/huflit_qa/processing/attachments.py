from pathlib import Path

ANCHOR_TEXTS = ["Bấm vào để xem", "vui lòng xem tại đây"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def find_attachment_file(attachments_base: Path, doc_id: int, index: int):
    """Tìm file attachment trong attachments_base/{doc_id}_*/.

    Returns:
        (path, "image") cho file ảnh
        (path, "txt")   cho file .txt
        (None, None)    nếu không tìm thấy
    """
    folders = list(attachments_base.glob(f"{doc_id}_*"))
    if not folders:
        return None, None

    folder = folders[0]
    image_file = None
    txt_file = None
    for f in folder.glob(f"{doc_id}_{index}__*"):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            image_file = f
            break
        if f.suffix.lower() == ".txt" and txt_file is None:
            txt_file = f

    if image_file:
        return image_file, "image"
    if txt_file:
        return txt_file, "txt"
    return None, None


def resolve_attachment_title(attachment_text: str, index: int, doc_content: str) -> str:
    """Trả về title rõ ràng cho attachment.

    - Text không mơ hồ: trả về nguyên
    - Text mơ hồ: parse từ doc_content
    - Fallback: "Tài liệu đính kèm {index}"
    """
    text = attachment_text.strip()
    if text not in ANCHOR_TEXTS:
        return text

    anchor = ANCHOR_TEXTS[0]
    for a in ANCHOR_TEXTS:
        if a in doc_content:
            anchor = a
            break
    segments = doc_content.split(anchor)

    if index <= 0 or index > len(segments) - 1:
        return f"Tài liệu đính kèm {index}"

    raw_lines = segments[index - 1].strip().splitlines()
    lines = []
    for ln in raw_lines:
        ln = ln.strip()
        if ln:
            lines.append(ln)
    if lines:
        return lines[-1]
    return f"Tài liệu đính kèm {index}"


def read_txt_content(path: Path):
    """Đọc nội dung file .txt.

    Returns:
        (content, "extracted") nếu có nội dung
        (None, "empty")        nếu rỗng hoặc không tồn tại
    """
    if not path.exists():
        return None, "empty"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None, "empty"
    return text, "extracted"
