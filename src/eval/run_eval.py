import argparse
from typing import List, Tuple

from src.observability.phoenix_client import setup_phoenix, shutdown_phoenix
from src.rag.generator import generate_answer
from src.rag.retriever import QdrantRetriever

DEFAULT_QUESTIONS = [
    "What is LangChain?",
    "What is LangGraph?",
    "How do I build an agent with LangGraph?",
]

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run sample evaluation queries against local RAG bot.")
    p.add_argument("--question", action="append", default=[], help="Add a question (can be repeated).")
    p.add_argument("--top-k", type=int, default=None, help="Override Qdrant retriever top-k.")
    return p

def main() -> None:
    args = build_argparser().parse_args()
    questions: List[str] = args.question or DEFAULT_QUESTIONS

    setup_phoenix()
    try:
        _run(questions, top_k=args.top_k)
    finally:
        shutdown_phoenix()


def _run(questions: List[str], *, top_k: int | None) -> None:
    retriever = QdrantRetriever(top_k=top_k)
    history: List[Tuple[str, str]] = []
    for q in questions:
        retrieved = retriever.retrieve(q)
        context = retrieved.format_for_prompt()
        result = generate_answer(
            question=q,
            context=context,
            history=history,
            language=None,
        )
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
