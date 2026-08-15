from typing import Any, Protocol

from src.models import SearchResult
from src.retriever import format_evidence

INSUFFICIENT_EVIDENCE = "知识库中没有足够信息回答这个问题。"

SYSTEM_PROMPT = f"""你是技术知识库助手。
只根据用户消息中提供的资料回答问题。
每个关键结论必须标注资料编号，例如 [1]。
资料不足时回答{INSUFFICIENT_EVIDENCE}。
不得补写资料中不存在的事实。"""


class ChatCompletionsAPI(Protocol):
    def create(self, **kwargs: object) -> Any:
        ...


class AnswerGenerator:
    def __init__(
        self,
        api: ChatCompletionsAPI,
        model: str,
    ) -> None:
        self._api = api
        self._model = model

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        if not results:
            return INSUFFICIENT_EVIDENCE

        content = (
            f"资料：\n{format_evidence(results)}"
            f"\n\n问题：{question}"
        )

        response = self._api.create(
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )

        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            raise ValueError("chat model returned empty content")

        return answer.strip()
