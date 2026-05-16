from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.utils import detect_language


@dataclass(frozen=True)
class GenerationResult:
    text: str


def _build_prompt_template(language: str) -> ChatPromptTemplate:
    """
    Build a ChatPromptTemplate compatible with the previous string prompt behavior.
    """
    if language == "ru":
        system = (
            "Ты помощник по документации LangChain и LangGraph. Отвечай на русском. "
            "Используй только предоставленный контекст документации. "
            "Если в контексте нет ответа, скажи, что сведений недостаточно."
        )
        user = (
            "История диалога:\n{history}\n\n"
            "Контекст документации:\n{context}\n\n"
            "Вопрос пользователя:\n{question}\n\n"
            "Ответ:"
        )
    else:
        system = (
            "You are a LangChain and LangGraph documentation assistant. Answer in English. "
            "Use only the provided documentation context. "
            "If the answer is not present in the context, say that the information is not available."
        )
        user = (
            "Conversation history:\n{history}\n\n"
            "Documentation context:\n{context}\n\n"
            "User question:\n{question}\n\n"
            "Answer:"
        )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user),
        ]
    )


def _format_history(history: List[Tuple[str, str]]) -> str:
    if not history:
        return ""
    return "\n".join([f"User: {u}\nAssistant: {a}" for (u, a) in history[-4:]])


def _message_to_openai_role(msg: Any) -> str:
    t = getattr(msg, "type", None) or msg.__class__.__name__.lower()
    if t == "system":
        return "system"
    if t in ("human", "user"):
        return "user"
    if t in ("ai", "assistant"):
        return "assistant"
    return "user"


def _chat_completions(
    *,
    url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    api_key: Optional[str],
    timeout_s: float,
) -> str:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    body["model"] = model

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response missing choices")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise RuntimeError("LLM response missing message content")
    return str(content).strip()


def generate_answer(
    *,
    question: str,
    context: str,
    history: List[Tuple[str, str]],
    language: Optional[str] = None,
) -> GenerationResult:
    """
    Call an OpenAI-compatible chat completions endpoint (e.g. LM Studio) to produce an answer.
    """
    cfg = settings()
    lang = language or detect_language(question)
    prompt = _build_prompt_template(language=lang)

    history_str = _format_history(history)
    if cfg.llm_max_input_tokens and len(context) > cfg.llm_max_input_tokens * 4:
        context = context[-cfg.llm_max_input_tokens * 4 :]

    lc_messages = prompt.format_messages(
        question=question,
        context=context,
        history=history_str,
    )
    messages = [
        {"role": _message_to_openai_role(m), "content": m.content}
        for m in lc_messages
    ]

    text = _chat_completions(
        url=cfg.llm_chat_completions_url,
        model=cfg.llm_model,
        messages=messages,
        max_tokens=cfg.llm_max_new_tokens,
        temperature=cfg.llm_temperature,
        api_key=cfg.llm_api_key,
        timeout_s=cfg.llm_request_timeout_s,
    )
    return GenerationResult(text=text)
