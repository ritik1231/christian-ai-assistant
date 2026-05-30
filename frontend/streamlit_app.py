"""
Christianity AI Assistant — Streamlit frontend.
Calls the pipeline directly (no HTTP round-trip needed for local demo).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # telemetry patch

import streamlit as st

st.set_page_config(
    page_title="Christian AI Assistant",
    page_icon="✝",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.config import DENOMINATIONS


@st.cache_resource(show_spinner=False)
def _ensure_index():
    """Build the ChromaDB index on first run; cached for the process lifetime."""
    from scripts.build_index import build_if_needed
    built = build_if_needed()
    return built


def _init():
    with st.spinner("Setting up scripture index (first run only — about 2 minutes)…"):
        _ensure_index()


from app.services.pipeline import run  # noqa: E402 — must import after index is ready
from app.services.memory import clear_session  # noqa: E402


def _show_image(url: str) -> None:
    """Render image via browser-side fetch (avoids server network restrictions)."""
    st.markdown(
        f'<img src="{url}" style="width:100%; border-radius:8px; margin-top:8px;" '
        f'onerror="this.style.display=\'none\'; this.nextSibling.style.display=\'block\'"/>'
        f'<div class="warn-badge" style="display:none">⚠️ Image failed to load. '
        f'<a href="{url}" target="_blank">Open in browser</a></div>',
        unsafe_allow_html=True,
    )

_init()

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Softer background */
.stApp { background-color: #f7f4f0; }

/* Ensure body text is always dark and readable */
.stApp, .stApp p, .stApp span, .stApp div {
    color: #1a1a1a;
}

/* Chat bubbles — white card with dark text for contrast on any background */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 8px;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    padding: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* Force all text inside chat messages to be dark */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #1a1a1a !important;
}

/* Mobile: tighten padding */
@media (max-width: 640px) {
    .stApp { padding: 0 4px; }
    [data-testid="stChatMessage"] { padding: 8px; }
}

/* Source pill */
.source-pill {
    display: inline-block;
    background: #e8eaf6;
    color: #283593;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px 3px 2px 0;
    font-weight: 500;
}

/* Warning badge */
.warn-badge {
    background: #fff3e0;
    color: #e65100;
    border-left: 3px solid #e65100;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.82rem;
    margin-top: 6px;
}

/* Block badge */
.block-badge {
    background: #fce4ec;
    color: #b71c1c;
    border-left: 3px solid #b71c1c;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.82rem;
    margin-top: 6px;
}

/* Provider chip */
.provider-chip {
    font-size: 0.70rem;
    color: #757575;
    margin-top: 4px;
}

/* Sidebar header */
.sidebar-section {
    font-size: 0.75rem;
    font-weight: 600;
    color: #9e9e9e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 16px 0 4px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "denomination" not in st.session_state:
    st.session_state.denomination = DENOMINATIONS[0]

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✝ Christian AI")
    st.markdown("Scripture-grounded · Denomination-aware · Safe")
    st.divider()

    st.markdown('<div class="sidebar-section">Tradition</div>', unsafe_allow_html=True)
    selected_denom = st.selectbox(
        "Your denomination",
        DENOMINATIONS,
        index=DENOMINATIONS.index(st.session_state.denomination),
        label_visibility="collapsed",
    )
    if selected_denom != st.session_state.denomination:
        st.session_state.denomination = selected_denom
        # Update denomination in existing session
        if st.session_state.session_id:
            from app.services.memory import update_denomination
            update_denomination(st.session_state.session_id, selected_denom)

    st.divider()
    st.markdown('<div class="sidebar-section">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
- 📖 Answer theology questions
- 📜 Cite & verify Bible verses
- 🖼 Generate Christian images
- 🛡 Block hallucinated scripture
- ⚖️ Handle contested theology
    """)

    st.divider()
    st.markdown('<div class="sidebar-section">Try asking</div>', unsafe_allow_html=True)
    examples = [
        "What does John 3:16 mean?",
        "Explain predestination vs free will",
        "Generate an image of the nativity",
        "What does the Bible say about anxiety?",
        "How should I pray?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        if st.session_state.session_id:
            clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    if st.session_state.session_id:
        st.markdown(
            f'<div class="provider-chip">Session: '
            f'{st.session_state.session_id[:8]}… | '
            f'{len(st.session_state.messages)} messages</div>',
            unsafe_allow_html=True,
        )

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("## ✝ Christian AI Assistant")
st.caption(
    f"Tradition: **{st.session_state.denomination}** · "
    "Powered by Groq LLaMA 3.3 · Bible: KJV (31,100 verses)"
)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙏" if msg["role"] == "assistant" else "👤"):
        if msg.get("is_image") and msg.get("image_url"):
            st.markdown(msg["content"])
            _show_image(msg["image_url"])
        else:
            st.markdown(msg["content"])

        # Blocked badge
        if msg.get("blocked"):
            st.markdown(
                f'<div class="block-badge">🚫 Blocked · {msg.get("block_category", "")}</div>',
                unsafe_allow_html=True,
            )

        # Unverified citations warning
        if msg.get("unverified"):
            refs = ", ".join(msg["unverified"])
            st.markdown(
                f'<div class="warn-badge">⚠️ Unverified citation(s): {refs}</div>',
                unsafe_allow_html=True,
            )

        # Output flagged warning
        if msg.get("output_flagged"):
            st.markdown(
                '<div class="warn-badge">⚠️ Response was reviewed and a safer reply was provided.</div>',
                unsafe_allow_html=True,
            )

        # Scripture sources
        if msg.get("sources"):
            with st.expander(f"📖 {len(msg['sources'])} scripture source(s)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<span class="source-pill">{src["ref"]}</span> '
                        f'<span style="font-size:0.85rem; color:#424242;">{src["text"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

        # Provider chip
        if msg.get("provider") and msg["provider"] not in ("", "blocked"):
            st.markdown(
                f'<div class="provider-chip">via {msg["provider"]}</div>',
                unsafe_allow_html=True,
            )

# ── Chat input ────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input(
    "Ask a scripture question or say 'generate an image of…'",
    key="chat_input",
) or prefill

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Call pipeline
    with st.chat_message("assistant", avatar="🙏"):
        with st.spinner("Searching scripture…"):
            result = run(
                user_message=user_input,
                session_id=st.session_state.session_id,
                denomination=st.session_state.denomination,
            )
            st.session_state.session_id = result["session_id"]

        # Render response
        if result.get("is_image") and result.get("image_url"):
            st.markdown(result["reply"])
            _show_image(result["image_url"])
        else:
            st.markdown(result["reply"])

        if result.get("blocked"):
            st.markdown(
                f'<div class="block-badge">🚫 Blocked · {result.get("block_category", "")}</div>',
                unsafe_allow_html=True,
            )

        if result.get("unverified"):
            refs = ", ".join(result["unverified"])
            st.markdown(
                f'<div class="warn-badge">⚠️ Unverified citation(s): {refs}</div>',
                unsafe_allow_html=True,
            )

        if result.get("output_flagged"):
            st.markdown(
                '<div class="warn-badge">⚠️ Response was reviewed and a safer reply was provided.</div>',
                unsafe_allow_html=True,
            )

        if result.get("sources"):
            with st.expander(f"📖 {len(result['sources'])} scripture source(s)", expanded=False):
                for src in result["sources"]:
                    st.markdown(
                        f'<span class="source-pill">{src["ref"]}</span> '
                        f'<span style="font-size:0.85rem; color:#424242;">{src["text"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("")

        if result.get("provider") and result["provider"] not in ("", "blocked"):
            st.markdown(
                f'<div class="provider-chip">via {result["provider"]}</div>',
                unsafe_allow_html=True,
            )

    # Save to display history
    st.session_state.messages.append({
        "role":           "assistant",
        "content":        result["reply"],
        "sources":        result.get("sources", []),
        "unverified":     result.get("unverified", []),
        "blocked":        result.get("blocked", False),
        "block_category": result.get("block_category", ""),
        "output_flagged": result.get("output_flagged", False),
        "is_image":       result.get("is_image", False),
        "image_url":      result.get("image_url"),
        "provider":       result.get("provider", ""),
    })
