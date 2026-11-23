"""Vietnamese prompts. The token budget covers the whole chat-formatted prompt (system + user turn)."""

from __future__ import annotations

from ..datasets import Question

SYSTEM = (
    "Bạn là trợ lý pháp lý. Chỉ sử dụng các điều luật được cung cấp để trả lời. "
    "Mỗi nhận định phải kèm trích dẫn căn cứ theo đúng dạng [Tên văn bản, Điều X, Khoản Y] "
    "(bỏ phần Khoản nếu trích cả Điều), với tên văn bản ghi đúng như trong ngoặc vuông ở đầu mỗi điều luật. "
    "Không trích dẫn điều luật không có trong phần được cung cấp."
)

INSTRUCTIONS = {
    "Đúng/Sai": "Dòng đầu tiên ghi \"Đáp án: Đúng\" hoặc \"Đáp án: Sai\". Sau đó giải thích ngắn gọn kèm trích dẫn.",
    "Trắc nghiệm": "Dòng đầu tiên ghi \"Đáp án: \" và một chữ cái (A, B, C hoặc D). Sau đó giải thích ngắn gọn kèm trích dẫn.",
    "Tự luận": "Dòng đầu tiên ghi \"Đáp án: \" và câu trả lời ngắn gọn nhất có thể. Sau đó giải thích ngắn gọn kèm trích dẫn.",
    "": "Trả lời đầy đủ, rõ ràng bằng tiếng Việt, kèm trích dẫn cho từng nhận định.",
}


def user_message(q: Question, context: str) -> str:
    parts = ["Các điều luật được cung cấp:", context.strip() or "(không có)", "", f"Câu hỏi: {q.text.strip()}"]
    if q.choices:
        parts += [f"{k}. {v}" for k, v in sorted(q.choices.items())]
    parts += ["", INSTRUCTIONS.get(q.qtype, INSTRUCTIONS[""])]
    return "\n".join(parts)


def messages(q: Question, context: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(q, context)}]


class PromptBuilder:
    """Applies the generator's chat template and measures prompts with the generator's tokenizer."""

    def __init__(self, tokenizer, enable_thinking: bool | None = None):
        self.tok = tokenizer
        self.kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}

    def render(self, q: Question, context: str) -> str:
        return self.tok.apply_chat_template(messages(q, context), tokenize=False, add_generation_prompt=True,
                                            **self.kwargs)

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def truncate(self, text: str, n: int) -> str:
        ids = self.tok(text, add_special_tokens=False)["input_ids"][:max(n, 0)]
        return self.tok.decode(ids)

    def context_budget(self, q: Question, budget: int) -> int:
        """Tokens left for the context once the template, instructions and question are paid for."""
        return budget - self.count(self.render(q, ""))
