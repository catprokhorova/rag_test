from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

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


def build_llama_cpp() -> LlamaCpp:
    cfg = settings()
    if cfg.llm_backend != "llama_cpp":
        raise RuntimeError(
            "Only llama_cpp backend is supported for the LangChain LlamaCpp adapter."
        )
    if not cfg.llm_gguf_path:
        raise RuntimeError("LLM_GGUF_PATH must be set for llama_cpp backend.")

    return LlamaCpp(
        model_path=cfg.llm_gguf_path,
        n_ctx=cfg.llm_ctx,
        n_threads=cfg.llm_threads,
        n_gpu_layers=0,
        temperature=cfg.llm_temperature,
        repeat_penalty=cfg.llm_repetition_penalty,
        max_tokens=cfg.llm_max_new_tokens,
        verbose=False,
    )


def build_chat_chain(*, language: str) -> Runnable:
    """
    Build a prompt → LlamaCpp chain.

    Retrieval is performed separately so we can keep the existing
    RetrievedContext formatting and source handling.
    """
    prompt = _build_prompt_template(language=language)
    llm = build_llama_cpp()
    return prompt | llm


def generate_answer(
    *,
    question: str,
    context: str,
    history: List[Tuple[str, str]],
    language: Optional[str] = None,
) -> GenerationResult:
    """
    High-level helper that mirrors LocalTextGenerator.generate API
    but uses a LangChain LlamaCpp chain under the hood.
    """
    cfg = settings()
    lang = language or detect_language(question)
    chain = build_chat_chain(language=lang)

    history_str = _format_history(history)
    # Keep rough parity with previous character-based truncation on context size.
    if cfg.llm_max_input_tokens and len(context) > cfg.llm_max_input_tokens * 4:
        context = context[-cfg.llm_max_input_tokens * 4 :]

    result = chain.invoke(
        {
            "question": question,
            "context": context,
            "history": history_str,
        }
    )
    text = str(result).strip()
    return GenerationResult(text=text)
