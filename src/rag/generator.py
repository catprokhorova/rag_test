import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langfuse import get_client, observe

from src.config import settings
from src.utils import detect_language


@dataclass(frozen=True)
class GenerationResult:
    text: str
    raw: str


def _parse_answer_from_llm(content: str) -> str:
    """
    Return user-facing answer text from LLM output.

    Supports {"answer": "..."} JSON (optionally wrapped in markdown fences).
    Falls back to the full string when parsing fails.
    """
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"]).strip()
    except json.JSONDecodeError:
        pass

    match = re.search(
        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text,
        flags=re.DOTALL,
    )
    if match:
        return json.loads(f'"{match.group(1)}"').strip()

    return content.strip()


def _build_prompt_template(language: str) -> ChatPromptTemplate:
    """
    Build a ChatPromptTemplate compatible with the previous string prompt behavior.
    """
    if language == "ru":
        system = (
            "Ты помощник по документации LangChain и LangGraph. Отвечай на русском. "
            "Используй только предоставленный контекст документации. "
            "Если в контексте нет ответа, скажи, что сведений недостаточно. "
            "Отвечай только валидным JSON одним объектом с единственным ключом \"answer\". "
            "Значение answer — связный текст для пользователя без chunk_id, ссылок на фрагменты "
            "и номеров вида [1]."
        )
        user = (
            "История диалога:\n{history}\n\n"
            "Контекст документации:\n{context}\n\n"
            "Вопрос пользователя:\n{question}"
        )
    else:
        system = (
            "You are a LangChain and LangGraph documentation assistant. Answer in English. "
            "Use only the provided documentation context. "
            "If the answer is not present in the context, say that the information is not available. "
            "Reply with valid JSON only: a single object with the key \"answer\". "
            "The answer value must be user-facing prose with no chunk_id values, "
            "citations, or bracketed reference markers like [1]."
        )
        user = (
            "Conversation history:\n{history}\n\n"
            "Documentation context:\n{context}\n\n"
            "User question:\n{question}"
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


@observe(name="llm-generation", as_type="generation", capture_input=False, capture_output=False)
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
    langfuse = get_client()
    langfuse.update_current_generation(
        input=messages,
        model=model,
        metadata={"endpoint": url},
    )

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

    usage = data.get("usage") or {}
    usage_details: Dict[str, int] = {}
    if usage.get("prompt_tokens") is not None:
        usage_details["input"] = int(usage["prompt_tokens"])
    if usage.get("completion_tokens") is not None:
        usage_details["output"] = int(usage["completion_tokens"])

    langfuse.update_current_generation(
        output=content,
        usage_details=usage_details or None,
    )
    return str(content).strip()


@observe(name="generate-answer", as_type="span", capture_input=False, capture_output=False)
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
    get_client().update_current_span(
        input={"question": question, "language": lang},
        metadata={"context_chars": len(context), "history_turns": len(history)},
    )
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

    raw = _chat_completions(
        url=cfg.llm_chat_completions_url,
        model=cfg.llm_model,
        messages=messages,
        max_tokens=cfg.llm_max_new_tokens,
        temperature=cfg.llm_temperature,
        api_key=cfg.llm_api_key,
        timeout_s=cfg.llm_request_timeout_s,
    )
    result = GenerationResult(text=_parse_answer_from_llm(raw), raw=raw)
    get_client().update_current_span(output={"answer": result.text})
    return result
