import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langfuse import Langfuse


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def get_tracing_client() -> "Langfuse":
    from langfuse import get_client

    return get_client()


def shutdown_tracing() -> None:
    if not langfuse_enabled():
        return
    get_tracing_client().shutdown()
