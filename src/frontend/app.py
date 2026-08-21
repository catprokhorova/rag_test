"""Streamlit chat UI for the Docs RAG backend facade."""

from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
CHAT_TIMEOUT_S = float(os.getenv("FRONTEND_CHAT_TIMEOUT_S", "120"))

MUSTARD = "#FFBF00"
MUSTARD_DARK = "#FFBF00"
GRAY_BG = "#F3F2EF"
GRAY_PANEL = "#E8E6E0"
INK = "#2B2B28"

EXAMPLE_PROMPTS = (
    "What is LangChain?",
    "How do document loaders work?",
    "How do I create a custom prompt template?",
    "What is LLM caching in LangChain?",
)

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {GRAY_BG};
    }}
    [data-testid="stSidebar"] {{
        background-color: {GRAY_PANEL};
        border-right: 3px solid {MUSTARD};
    }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {INK};
    }}
    .app-banner {{
        background: linear-gradient(90deg, {MUSTARD} 0%, #d4b44a 55%, {GRAY_PANEL} 100%);
        border-radius: 14px;
        padding: 1.15rem 1.4rem 1.05rem;
        margin-bottom: 1.1rem;
        color: {INK};
        box-shadow: 0 8px 20px rgba(43, 43, 40, 0.08);
    }}
    .app-banner h1 {{
        font-size: 1.55rem;
        margin: 0 0 0.25rem 0;
        letter-spacing: 0.01em;
    }}
    .app-banner p {{
        margin: 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }}
    .health-ok {{
        color: {MUSTARD_DARK};
        font-weight: 600;
    }}
    .health-bad {{
        color: #8a3a2a;
        font-weight: 600;
    }}
    .session-chip {{
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.75rem;
        background: {GRAY_BG};
        border: 1px solid {MUSTARD};
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        display: inline-block;
        color: {INK};
        word-break: break-all;
    }}
    [data-testid="stChatMessage"] {{
        background: #fff;
        border: 1px solid #ddd9d0;
        border-radius: 12px;
        padding: 0.35rem 0.5rem;
    }}
    div[data-testid="stChatInput"] textarea {{
        background: #fff;
    }}
    .stButton > button {{
        background-color: {MUSTARD};
        color: {INK};
        border: none;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: {MUSTARD_DARK};
        color: #fff;
        border: none;
    }}
    header[data-testid="stHeader"] {{
        background: transparent;
    }}
</style>
"""


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "language" not in st.session_state:
        st.session_state.language = "auto"


def _reset_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def _backend_health() -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            return True, "Backend is reachable"
        return False, f"Health check failed ({response.status_code})"
    except httpx.RequestError as exc:
        return False, f"Cannot reach backend: {exc}"


def _send_chat(message: str) -> tuple[str, list[dict]]:
    payload = {
        "session_id": st.session_state.session_id,
        "message": message,
        "language": st.session_state.language,
    }
    with httpx.Client(timeout=CHAT_TIMEOUT_S) as client:
        response = client.post(f"{BACKEND_URL}/chat", json=payload)
    if response.status_code != 200:
        detail = response.text
        raise RuntimeError(f"Chat failed ({response.status_code}): {detail}")
    data = response.json()
    st.session_state.session_id = data.get("session_id") or st.session_state.session_id
    return data["answer"], data.get("sources") or []


def _unique_sources(sources: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[tuple] = set()
    for source in sources:
        key = (
            source.get("title") or "",
            source.get("source_file") or "",
            source.get("page"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _source_label(source: dict) -> str:
    label = source.get("source_file") or source.get("title") or "Source"
    page = source.get("page")
    if page is not None:
        try:
            # PagedPDFSplitter stores 0-based page numbers.
            label = f"{label}, p. {int(page) + 1}"
        except (TypeError, ValueError):
            pass
    return label


def _render_sources(sources: list[dict]) -> None:
    unique = _unique_sources(sources)
    if not unique:
        return
    lines = ["**Sources**"]
    for source in unique:
        label = _source_label(source)
        url = (source.get("url") or "").strip()
        if url.startswith(("http://", "https://")):
            lines.append(f"- [{label}]({url})")
        else:
            lines.append(f"- {label}")
    st.markdown("\n".join(lines))


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Docs RAG")
        st.caption("Local chatbot over ingested LangChain PDFs.")

        healthy, health_msg = _backend_health()
        badge = "health-ok" if healthy else "health-bad"
        st.markdown(f'<p class="{badge}">{health_msg}</p>', unsafe_allow_html=True)
        st.caption(f"API: `{BACKEND_URL}`")

        st.markdown("#### Answer language")
        st.selectbox(
            "Language",
            options=["auto", "en", "ru"],
            key="language",
            label_visibility="collapsed",
        )

        st.markdown("#### Conversation")
        st.markdown(
            f'<span class="session-chip">{st.session_state.session_id}</span>',
            unsafe_allow_html=True,
        )
        if st.button("New conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()

        st.markdown("#### Try asking")
        for prompt in EXAMPLE_PROMPTS:
            if st.button(prompt, key=f"ex-{prompt}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()


def _consume_prompt() -> str | None:
    pending = st.session_state.pop("pending_prompt", None)
    typed = st.chat_input("Ask about LangChain documentation…")
    return pending or typed


def main() -> None:
    st.set_page_config(
        page_title="Docs RAG Chat",
        page_icon="●",
        layout="centered",
    )
    _init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="app-banner">
            <h1>LangChain docs chat</h1>
            <p>Ask in English or Russian. Answers are generated from the ingested PDFs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_sidebar()

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                _render_sources(item.get("sources") or [])

    prompt = _consume_prompt()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    sources: list[dict] = []
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating an answer…"):
            try:
                answer, sources = _send_chat(prompt)
            except Exception as exc:
                answer = f"Sorry, the chat request failed.\n\n`{exc}`"
        st.markdown(answer)
        _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
