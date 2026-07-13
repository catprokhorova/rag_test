import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import openai
from langchain_core.prompts import ChatPromptTemplate
from langfuse import get_client, observe

from src.config import settings
from src.observability.phoenix_client import chain_span, llm_span
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


def _split_instructions_and_input(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """Map chat-style messages to Responses API instructions + input."""
    instructions_parts: List[str] = []
    input_parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            instructions_parts.append(content)
        else:
            input_parts.append(content)
    return "\n\n".join(instructions_parts).strip(), "\n\n".join(input_parts).strip()


@llm_span("llm-generation")
@observe(name="llm-generation", as_type="generation", capture_input=False, capture_output=False)
def _yandex_responses(
    *,
    folder: str,
    model: str,
    base_url: str,
    messages: List[Dict[str, str]],
    max_output_tokens: int,
    temperature: float,
    api_key: str,
    timeout_s: float,
) -> str:
    model_uri = f"gpt://{folder}/{model}"
    instructions, input_text = _split_instructions_and_input(messages)

    langfuse = get_client()
    langfuse.update_current_generation(
        input=messages,
        model=model_uri,
        metadata={"endpoint": base_url, "folder": folder},
    )

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        project=folder,
        timeout=timeout_s,
    )
    response = client.responses.create(
        model=model_uri,
        temperature=temperature,
        instructions=instructions,
        input=input_text,
        max_output_tokens=max_output_tokens,
    )

    content = response.output_text
    if content is None:
        raise RuntimeError("LLM response missing output_text")

    usage = getattr(response, "usage", None)
    usage_details: Dict[str, int] = {}
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is not None:
            usage_details["input"] = int(input_tokens)
        if output_tokens is not None:
            usage_details["output"] = int(output_tokens)

    langfuse.update_current_generation(
        output=content,
        usage_details=usage_details or None,
    )
    return str(content).strip()


@chain_span("generate-answer")
@observe(name="generate-answer", as_type="span", capture_input=False, capture_output=False)
def generate_answer(
    *,
    question: str,
    context: str,
    history: List[Tuple[str, str]],
    language: Optional[str] = None,
) -> GenerationResult:
    """
    Call Yandex Cloud AI Studio (Responses API) to produce an answer.
    """
    cfg = settings()
    if not cfg.yandex_cloud_api_key:
        raise RuntimeError("YANDEX_CLOUD_API_KEY is not set")

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
        {
            "role": "system" if getattr(m, "type", None) == "system" else "user",
            "content": m.content,
        }
        for m in lc_messages
    ]

    raw = _yandex_responses(
        folder=cfg.yandex_cloud_folder,
        model=cfg.yandex_cloud_model,
        base_url=cfg.yandex_cloud_base_url,
        messages=messages,
        max_output_tokens=cfg.llm_max_new_tokens,
        temperature=cfg.llm_temperature,
        api_key=cfg.yandex_cloud_api_key,
        timeout_s=cfg.llm_request_timeout_s,
    )
    result = GenerationResult(text=_parse_answer_from_llm(raw), raw=raw)
    get_client().update_current_span(output={"answer": result.text})
    return result
