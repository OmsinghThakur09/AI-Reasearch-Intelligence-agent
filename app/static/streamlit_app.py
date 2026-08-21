# app/static/streamlit_app

"""
Streamlit frontend for AI Research agent

- Streams the answer token-by-token from POST /research/stream (SSE).
- Keeps ONE session_id for the whole browser session (stored in
  st.session_state), so every follow-up question in this chat reuses it and
  the backend's one-turn-back history (SESSION_LAST_QA) and ChromaDB
  session filter both stay consistent.
- Shows full chat history locally (the backend itself only remembers the
  last turn).
"""

import json
import time
import requests

import streamlit as st
from urllib.parse import urlparse

API_URL = "http://localhost:8000"

TOKEN_REVEAL_DELAY = 0.010  # to delay streaming tokens

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

# styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #0f1419;
    --surface: #171d26;
    --surface-2: #1c2430;
    --border: #2a3341;
    --text: #e8eaed;
    --text-muted: #8b93a1;
    --accent: #c9a15a;
    --accent-soft: rgba(201, 161, 90, 0.12);
    --accent-2: #5fb8a8;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
section.main > div.block-container { padding-top: 2.2rem; max-width: 760px; }

/* hide default streamlit chrome for a cleaner, product-like shell */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* title bar */
.app-title { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.15rem; }
.app-title .mark { font-size: 1.4rem; }
.app-title h1 {
    font-family: 'Source Serif 4', serif; font-weight: 600;
    font-size: 1.4rem; color: var(--text); margin: 0;
}
.app-sub { color: var(--text-muted); font-size: 0.84rem; margin-bottom: 1.5rem; }

/* landing / empty-state hero */
.landing-wrap { text-align: center; padding: 4rem 0 2.2rem 0; }
.landing-wrap h2 {
    font-family: 'Source Serif 4', serif; font-weight: 600;
    font-size: 2.2rem; color: var(--text); margin-bottom: 0.55rem;
}
.landing-wrap p { color: var(--text-muted); font-size: 0.95rem; margin: 0; }

/* chat bubbles */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.85rem;
}
[data-testid="stChatMessage"]:has([data-testid*="Assistant"]) { border-left: 3px solid var(--accent); }
[data-testid="stChatMessage"]:has([data-testid*="User"]) { background: var(--surface-2); }

/* "Thinking..." indicator shown before the first token arrives */
.thinking { display: flex; align-items: center; gap: 0.5rem; color: var(--text-muted); font-size: 0.92rem; }
.thinking .dots span {
    display: inline-block; width: 5px; height: 5px; margin-right: 3px;
    background: var(--accent); border-radius: 50%;
    animation: think-bounce 1.1s infinite ease-in-out;
}
.thinking .dots span:nth-child(2) { animation-delay: 0.15s; }
.thinking .dots span:nth-child(3) { animation-delay: 0.30s; }
@keyframes think-bounce {
    0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
    40% { opacity: 1; transform: translateY(-3px); }
}

/* sources, styled like footnote / citation chips */
.src-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--text-muted);
    letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.45rem;
}
.src-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    color: var(--accent); background: var(--accent-soft);
    border: 1px solid rgba(201, 161, 90, 0.35); border-radius: 6px;
    padding: 0.2rem 0.55rem; margin: 0.15rem 0.3rem 0.15rem 0;
}
.src-chip a { color: var(--accent); text-decoration: none; }
.src-chip a:hover { text-decoration: underline; }

/* chat input */
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 999px !important;
}
[data-testid="stChatInput"] { border-top: none; }

/* sidebar */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .stCode { font-family: 'IBM Plex Mono', monospace; }
</style>
""",
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # full chat history
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False  # True while a request is streaming
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None  # query queued by the input-lock handshake


def _domain(url: str) -> str:
    "best-effort short domain label for a source url, for the citation chips"
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else (netloc or url)
    except Exception:
        return url


def render_sources(sources: list[str]) -> None:
    "render a list of source urls as an expander of citation-style chips"
    with st.expander(f"Sources ({len(sources)})"):
        chips = "".join(
            f'<span class="src-chip">🔗 <a href="{s}" target="_blank">{_domain(s)}</a></span>'
            for s in sources
        )
        st.markdown(
            f'<div class="src-label">Referenced</div>{chips}', unsafe_allow_html=True
        )


def _start_new_session() -> None:
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.pending_query = None
    st.session_state.is_generating = False


# header
st.markdown(
    '<div class="app-title"><span class="mark">🔎</span><h1>AI Research Agent</h1></div>'
    '<div class="app-sub">Multi-source web research, synthesized and cited.</div>',
    unsafe_allow_html=True,
)


# landing hero, shown only before the first message
if not st.session_state.messages and not st.session_state.is_generating:
    st.markdown(
        '<div class="landing-wrap">'
        "<h2>What should we search today?</h2>"
        "<p>Ask a research question — I'll search the web, read the sources, and answer with citations.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

# replay existing chat history
for msg in st.session_state.messages:
    avatar = "🔎" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# chat input: disabled while a previous answer is still streaming
query = st.chat_input(
    (
        "Waiting for the current answer to finish..."
        if st.session_state.is_generating
        else "Ask a research question..."
    ),
    disabled=st.session_state.is_generating,
)

if query and not st.session_state.is_generating:
    # phase 1: lock the input immediately, then rerun.
    st.session_state.pending_query = query
    st.session_state.is_generating = True
    st.rerun()

# phase 2: a query is queued and the input is locked
if st.session_state.pending_query and st.session_state.is_generating:
    current_query = st.session_state.pending_query

    with st.chat_message("user", avatar="🧑"):
        st.markdown(current_query)

    with st.chat_message("assistant", avatar="🔎"):
        thinking = st.empty()
        thinking.markdown(
            '<div class="thinking"><span class="dots"><span></span><span></span><span></span></span>Thinking…</div>',
            unsafe_allow_html=True,
        )
        placeholder = st.empty()
        streamed_text = ""
        sources = []
        error_text = None
        first_token_seen = False

        payload = {"query": current_query, "session_id": st.session_state.session_id}

        try:
            with requests.post(
                f"{API_URL}/research/stream", json=payload, stream=True, timeout=100
            ) as req:
                req.raise_for_status()

                current_event = None
                for raw_line in req.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    if raw_line.startswith("event:"):
                        current_event = raw_line.split("event:", 1)[1].strip()
                        continue

                    if raw_line.startswith("data:"):
                        data_str = raw_line.split("data:", 1)[1].strip()

                        if current_event == "session":
                            data = json.loads(data_str)
                            st.session_state.session_id = data[
                                "session_id"
                            ]  # grab session id for follow-ups

                        elif current_event == "token":
                            if not first_token_seen:
                                first_token_seen = True
                                thinking.empty()

                            for ch in data_str:
                                streamed_text += ch
                                placeholder.markdown(streamed_text + "▌")
                                time.sleep(
                                    TOKEN_REVEAL_DELAY
                                )  # delaying token to look like typewriter

                        elif current_event == "done":
                            data = json.loads(data_str)
                            streamed_text = data["answer"]
                            sources = data.get("sources", [])
                            st.session_state.session_id = data.get(
                                "session_id", st.session_state.session_id
                            )

                        elif current_event == "error":
                            error_text = data_str

            thinking.empty()
            if error_text:
                placeholder.error(f"Something went wrong: {error_text}")

            else:
                placeholder.markdown(streamed_text)
                if sources:
                    render_sources(sources)

        except requests.exceptions.RequestException as e:
            error_text = str(e)
            thinking.empty()
            placeholder.error(f"Could not reach the research backend at {API_URL}: {e}")

    st.session_state.messages.append({"role": "user", "content": current_query})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": streamed_text if not error_text else f"⚠️ {error_text}",
            "sources": sources,
        }
    )

    # phase 3: unlock the input and rerun so the finished turn settles
    # into normal chat history and the box becomes usable again
    st.session_state.pending_query = None
    st.session_state.is_generating = False
    st.rerun()

# sidebar: session controls
with st.sidebar:
    st.markdown(
        '<div class="app-sub" style="margin-bottom:0.3rem;">SESSION</div>',
        unsafe_allow_html=True,
    )
    st.code(st.session_state.session_id or "not started yet")
    st.button(
        "＋ New session",
        on_click=_start_new_session,
        disabled=st.session_state.is_generating,
        use_container_width=True,
    )
