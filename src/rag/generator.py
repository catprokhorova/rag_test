from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.config import settings
from src.utils import detect_language


@dataclass(frozen=True)
class GenerationResult:
    text: str


class LocalTextGenerator:
    """
    Local text generation.

    Default backend is llama.cpp (GGUF) for low-resource CPU setups.
    """

    def __init__(self, *, model_name: Optional[str] = None):
        cfg = settings()
        self.backend = cfg.llm_backend
        self.model_name = model_name or cfg.llm_model
        self._llama = None

        if self.backend == "llama_cpp":
            from llama_cpp import Llama  # type: ignore

            self._llama = Llama(
                model_path=cfg.llm_gguf_path,
                n_ctx=cfg.llm_ctx,
                n_threads=cfg.llm_threads,
                n_gpu_layers=0,
                verbose=False,
            )
        elif self.backend == "hf_transformers":
            raise RuntimeError(
                "hf_transformers backend is not installed in this low-resource setup. "
                "Set LLM_BACKEND=llama_cpp and provide LLM_GGUF_PATH."
            )
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {self.backend}")

    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        history: List[Tuple[str, str]],
        language: str,
    ) -> str:
        if language == "ru":
            system = (
                "Ты помощник по Moodle. Отвечай на русском. "
                "Используй только предоставленный контекст документации. "
                "Если в контексте нет ответа, скажи, что сведений недостаточно."
            )
            history_block = "\n".join(
                [f"User: {u}\nAssistant: {a}" for (u, a) in history[-4:]]
            )
            if history_block:
                history_block = f"История диалога:\n{history_block}\n\n"
            return (
                f"{system}\n\n"
                f"{history_block}"
                f"Контекст документации:\n{context}\n\n"
                f"Вопрос пользователя:\n{question}\n\n"
                f"Ответ:"
            )

        system = (
            "You are a Moodle support assistant. Answer in English. "
            "Use only the provided documentation context. "
            "If the answer is not present in the context, say that the information is not available."
        )
        history_block = "\n".join([f"User: {u}\nAssistant: {a}" for (u, a) in history[-4:]])
        if history_block:
            history_block = f"Conversation history:\n{history_block}\n\n"
        return (
            f"{system}\n\n"
            f"{history_block}"
            f"Documentation context:\n{context}\n\n"
            f"User question:\n{question}\n\n"
            f"Answer:"
        )

    def generate(
        self,
        *,
        question: str,
        context: str,
        history: List[Tuple[str, str]],
        language: Optional[str] = None,
    ) -> GenerationResult:
        cfg = settings()
        lang = language or detect_language(question)
        prompt = self.build_prompt(question=question, context=context, history=history, language=lang)
        if self.backend != "llama_cpp" or self._llama is None:
            raise RuntimeError("LLM is not initialized (expected llama_cpp backend).")

        # llama.cpp runs with a fixed context (n_ctx). We keep prompt short by:
        # - limiting history elsewhere
        # - limiting retrieved context formatting
        # Additionally, allow user to control cfg.llm_max_input_tokens by truncating chars.
        if cfg.llm_max_input_tokens and len(prompt) > cfg.llm_max_input_tokens * 4:
            prompt = prompt[-cfg.llm_max_input_tokens * 4 :]

        out = self._llama.create_completion(
            prompt=prompt,
            max_tokens=cfg.llm_max_new_tokens,
            temperature=cfg.llm_temperature,
            repeat_penalty=cfg.llm_repetition_penalty,
            stop=["\n\nUser:", "\n\nВопрос пользователя:"],
        )
        text = (out.get("choices") or [{}])[0].get("text") or ""
        return GenerationResult(text=str(text).strip())

