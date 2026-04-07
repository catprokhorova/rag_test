from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.config import settings
from src.utils import detect_language


@dataclass(frozen=True)
class GenerationResult:
    text: str


class LocalTextGenerator:
    """
    Local text generation using a HuggingFace Transformers model.

    Works fully offline after the model is downloaded locally.
    """

    def __init__(self, *, model_name: Optional[str] = None):
        cfg = settings()
        self.model_name = model_name or cfg.llm_model
        self._tokenizer = None
        self._model = None
        self._is_encoder_decoder = None
        self._device = self._pick_device()
        self._load_model()

    def _pick_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        # Some models need explicit padding token.
        if not self._tokenizer.pad_token_id and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        tmp = None
        # is_encoder_decoder works across many model configs.
        # We'll inspect config to decide model class.
        model_config = AutoModelForSeq2SeqLM.from_config  # type: ignore[attr-defined]
        # Instead of clever introspection, try to load seq2seq first; fallback to causal.
        try:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._is_encoder_decoder = True
        except Exception:
            tmp = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model = tmp
            self._is_encoder_decoder = False

        self._model.to(self._device)
        self._model.eval()

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

    @torch.inference_mode()
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

        # Token budget: truncate input prompt.
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.llm_max_input_tokens,
        ).to(self._device)

        if self._is_encoder_decoder:
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=cfg.llm_max_new_tokens,
                temperature=cfg.llm_temperature,
                repetition_penalty=cfg.llm_repetition_penalty,
                do_sample=cfg.llm_temperature > 0,
            )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=cfg.llm_max_new_tokens,
                temperature=cfg.llm_temperature,
                repetition_penalty=cfg.llm_repetition_penalty,
                do_sample=cfg.llm_temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            # For causal models, decode returns full prompt + completion;
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            text = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()

        # Some models echo prompt; keep the last non-empty chunk.
        return GenerationResult(text=text.strip())

