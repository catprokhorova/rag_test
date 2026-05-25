"""Arize Phoenix (OpenTelemetry) integration for docs-rag."""

from __future__ import annotations

import os
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from openinference.instrumentation._tracer_providers import TracerProvider

F = TypeVar("F", bound=Callable[..., Any])

_tracer: Any = None


def phoenix_enabled() -> bool:
    return bool(os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip())


@lru_cache(maxsize=1)
def _register() -> Optional["TracerProvider"]:
    if not phoenix_enabled():
        return None
    from phoenix.otel import register

    project_name = os.getenv("PHOENIX_PROJECT_NAME", "docs-rag")
    auto_instrument = os.getenv("PHOENIX_AUTO_INSTRUMENT", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    protocol = os.getenv("PHOENIX_OTEL_PROTOCOL", "http/protobuf").strip().lower()
    if protocol not in {"http/protobuf", "grpc"}:
        protocol = "http/protobuf"
    return register(
        project_name=project_name,
        protocol=protocol,  # type: ignore[arg-type]
        batch=True,
        auto_instrument=auto_instrument,
    )


def setup_phoenix() -> None:
    """Initialize Phoenix OTEL from PHOENIX_COLLECTOR_ENDPOINT (and related env vars)."""
    _register()


def get_tracer():
    global _tracer
    provider = _register()
    if provider is None:
        return None
    if _tracer is None:
        _tracer = provider.get_tracer("docs-rag")
    return _tracer


def shutdown_phoenix() -> None:
    if not phoenix_enabled():
        return
    provider = _register()
    if provider is not None:
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    _register.cache_clear()
    global _tracer
    _tracer = None


def _lazy_oi_decorator(span_name: str, method: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not phoenix_enabled():
                return fn(*args, **kwargs)
            traced = getattr(wrapper, "_phoenix_traced", None)
            if traced is None:
                tracer = get_tracer()
                if tracer is None:
                    return fn(*args, **kwargs)
                traced = getattr(tracer, method)(name=span_name)(fn)
                wrapper._phoenix_traced = traced  # type: ignore[attr-defined]
            return wrapper._phoenix_traced(*args, **kwargs)  # type: ignore[attr-defined]

        return wrapper

    return decorator


def retriever_span(name: str = "retrieve-docs") -> Callable[[F], F]:
    """Trace vector retrieval as an OpenInference RETRIEVER span."""

    def decorator(fn: F) -> F:
        if not phoenix_enabled():
            return fn

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            if tracer is None:
                return fn(*args, **kwargs)

            from opentelemetry.trace import Status, StatusCode

            query = kwargs.get("query")
            if query is None and len(args) > 1:
                query = args[1]

            with tracer.start_as_current_span(
                name,
                openinference_span_kind="RETRIEVER",
            ) as span:
                if query is not None and hasattr(span, "set_input"):
                    span.set_input({"query": query})
                try:
                    result = fn(*args, **kwargs)
                    if hasattr(span, "set_output"):
                        span.set_output(
                            {
                                "num_chunks": len(getattr(result, "documents", []) or []),
                                "sources": getattr(result, "sources", lambda: [])(),
                            }
                        )
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception:
                    span.set_status(Status(StatusCode.ERROR))
                    raise

        return wrapper

    return decorator


def chain_span(name: str = "generate-answer") -> Callable[[F], F]:
    return _lazy_oi_decorator(name, "chain")


def llm_span(name: str = "llm-generation") -> Callable[[F], F]:
    return _lazy_oi_decorator(name, "llm")
