import argparse
from typing import List

from src.rag.generator import LocalTextGenerator
from src.rag.retriever import QdrantRetriever


DEFAULT_QUESTIONS = [
    "Как создать новый курс в Moodle?",
    "Как настроить систему оценок в Moodle?",
    "Как просмотреть журналы активности пользователей?",
]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run sample evaluation queries against local RAG bot.")
    p.add_argument("--question", action="append", default=[], help="Add a question (can be repeated).")
    p.add_argument("--top-k", type=int, default=None, help="Override Qdrant retriever top-k.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    questions: List[str] = args.question or DEFAULT_QUESTIONS

    retriever = QdrantRetriever(top_k=args.top_k)
    generator = LocalTextGenerator()

    history = []
    for q in questions:
        retrieved = retriever.retrieve(q)
        context = retrieved.format_for_prompt()
        result = generator.generate(question=q, context=context, history=history, language=None)
        history.append((q, result.text))

        print("=" * 80)
        print(f"Q: {q}")
        print("-" * 80)
        print(f"A: {result.text.strip()}")
        print("-" * 80)
        print("Sources:")
        for s in retrieved.sources():
            print(f"- {s['title']} ({s['url']}) [score={s['score']:.4f}]")


if __name__ == "__main__":
    main()

