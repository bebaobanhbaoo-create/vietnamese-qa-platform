"""Prompt for grounded, cited answers in Vietnamese."""

from huflit_qa.generation.llm import Message

# Fixed sentence so refusals can be detected exactly in evaluation.
REFUSAL = "Không tìm thấy thông tin này trong các văn bản hiện có."

SYSTEM_PROMPT = f"""Bạn là trợ lý tra cứu quy chế và thông báo của Trường Đại học Ngoại ngữ - Tin học \
TP. Hồ Chí Minh (HUFLIT), trả lời câu hỏi của sinh viên.

Quy tắc:
1. Chỉ dùng thông tin trong các tài liệu được đánh số [1], [2], ... bên dưới. Không dùng kiến thức bên ngoài.
2. Sau mỗi ý, ghi số tài liệu làm căn cứ trong ngoặc vuông, ví dụ: "... tối đa 25 tín chỉ [2]."
3. Nếu các tài liệu không đủ để trả lời, chỉ trả lời đúng một câu: "{REFUSAL}"
4. Nếu các tài liệu có nhiều phiên bản khác nhau của cùng một quy định, ưu tiên văn bản ban hành gần nhất \
và nêu rõ ngày ban hành; nếu quy định áp dụng khác nhau theo khóa, nói rõ khóa nào.
5. Trả lời ngắn gọn, chính xác, bằng tiếng Việt; giữ nguyên các con số, thời hạn, tên văn bản."""


def format_source(index: int, chunk: dict) -> str:
    title = " — ".join(p for p in (chunk["doc_title"], chunk.get("chunk_title")) if p)
    meta = []
    if chunk.get("issued_date"):
        meta.append(f"ban hành {chunk['issued_date']}")
    if chunk.get("date_posted"):
        meta.append(f"đăng {chunk['date_posted']}")
    if chunk.get("issuer_unit"):
        meta.append(chunk["issuer_unit"])
    header = f"[{index}] {title}" + (f" ({'; '.join(meta)})" if meta else "")
    return f"{header}\n{chunk['text'].strip()}"


def build_messages(question: str, chunks: list[dict]) -> list[Message]:
    """System prompt + one user message holding the numbered sources and the question."""
    sources = "\n\n".join(format_source(i, c) for i, c in enumerate(chunks, 1))
    user = f"Tài liệu:\n\n{sources}\n\nCâu hỏi: {question.strip()}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
