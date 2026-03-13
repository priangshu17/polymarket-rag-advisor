"""
app.py — Polymarket RAG Advisor
Main Streamlit UI (v2)
- Supports pre-built default KB + user-uploaded KB (independent vector stores)
- LLM provider toggle: Gemini 2.5 Flash-Lite (1 000 RPD) or Groq LLaMA 3.3 70B (500K tok/day)
- Answer mode: Concise vs Detailed
- Live Polymarket data + Tavily web search
"""

import os
import sys
import shutil

import streamlit as st

# ── path so local packages resolve regardless of working directory ────────────
sys.path.insert(0, os.path.dirname(__file__))

from config.config import (
    GEMINI_API_KEY,
    GROQ_API_KEY,
    TAVILY_API_KEY,
    VECTOR_STORE_PATH,
    DEFAULT_KB_PATH,
    LLM_PROVIDER,
)
from models.embeddings import VectorStore
from utils.file_ingestion import ingest_file
from utils.rag_pipeline import run_rag

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Polymarket RAG Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── base ── */
.stApp { background-color: #0e1117; color: #e2e8f0; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b27 0%, #0e1117 100%);
    border-right: 1px solid #1e2738;
}

/* ── answer mode badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}
.badge-concise  { background:#1d4ed8; color:#fff; }
.badge-detailed { background:#065f46; color:#fff; }

/* ── source snippet box ── */
.source-box {
    background:#0f172a;
    border: 1px solid #1e2d45;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
    line-height: 1.5;
}

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #1e2738;
    border-radius: 8px;
    padding: 8px;
}

/* ── KB status pill ── */
.kb-pill {
    display:inline-block;
    padding:3px 10px;
    border-radius:999px;
    font-size:11px;
    font-weight:600;
    margin:2px 0;
}
.kb-pill-on  { background:#064e3b; color:#6ee7b7; border:1px solid #10b981; }
.kb-pill-off { background:#1f2937; color:#6b7280; border:1px solid #374151; }

/* ── provider badge ── */
.provider-gemini { color:#4f8ef7; font-weight:700; }
.provider-groq   { color:#f59e0b; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading knowledge bases…")
def load_vector_stores():
    """Load both vector stores once and cache across reruns."""
    # User-uploaded KB
    user_vs = VectorStore()
    user_vs.load(VECTOR_STORE_PATH)

    # Default pre-built KB
    default_vs = VectorStore()
    loaded = default_vs.load(DEFAULT_KB_PATH)
    if not loaded:
        # Auto-build from sources if not yet built
        sources_dir = os.path.join(
            os.path.dirname(__file__), "knowledge_base", "sources"
        )
        if os.path.isdir(sources_dir):
            txt_files = sorted(
                f for f in os.listdir(sources_dir) if f.endswith(".txt")
            )
            for fname in txt_files:
                try:
                    with open(os.path.join(sources_dir, fname), "r", encoding="utf-8") as f:
                        text = f.read()
                    default_vs.add_documents([text], source_name=fname)
                except Exception:
                    pass
            if default_vs.total_chunks > 0:
                default_vs.save(DEFAULT_KB_PATH)

    return user_vs, default_vs


if "user_vs" not in st.session_state or "default_vs" not in st.session_state:
    u, d = load_vector_stores()
    st.session_state.user_vs    = u
    st.session_state.default_vs = d

if "chat_history"  not in st.session_state:
    st.session_state.chat_history = []

if "answer_mode"   not in st.session_state:
    st.session_state.answer_mode = "detailed"

if "llm_provider"  not in st.session_state:
    st.session_state.llm_provider = LLM_PROVIDER   # from config default


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Polymarket RAG Advisor")
    st.markdown("*AI-powered prediction market analysis*")
    st.divider()

    # ── LLM Provider ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 LLM Provider")
    provider = st.radio(
        label="",
        options=["gemini", "groq"],
        index=0 if st.session_state.llm_provider == "gemini" else 1,
        format_func=lambda x: (
            "✨ Gemini 2.5 Flash-Lite  (1 000 req/day free)"
            if x == "gemini"
            else "⚡ Groq LLaMA 3.3 70B  (500K tok/day free)"
        ),
    )
    st.session_state.llm_provider = provider
    # propagate into the live config so llm.py picks it up
    import config.config as _cfg
    _cfg.LLM_PROVIDER = provider

    with st.expander("ℹ️ Which provider should I use?"):
        st.markdown("""
| | Gemini Flash-Lite | Groq LLaMA 70B |
|---|---|---|
| Free req/day | **1 000** | ~unlimited* |
| Speed | Fast | Very fast |
| Quality | Excellent | Very good |
| Get key | [aistudio.google.com](https://aistudio.google.com) | [console.groq.com](https://console.groq.com) |

*Groq has a token/day limit (~500 K) rather than a request limit.
Use Groq for heavy testing, Gemini for production quality.
""")

    st.divider()

    # ── API Key Status ─────────────────────────────────────────────────────────
    st.markdown("### 🔑 API Status")
    gemini_ok = _cfg.GEMINI_API_KEY not in ("", "YOUR_GEMINI_API_KEY_HERE")
    groq_ok   = _cfg.GROQ_API_KEY   not in ("", "YOUR_GROQ_API_KEY_HERE")
    tavily_ok = _cfg.TAVILY_API_KEY not in ("", "YOUR_TAVILY_API_KEY_HERE")

    st.markdown(f"{'🟢' if gemini_ok else '🔴'} Gemini API Key")
    st.markdown(f"{'🟢' if groq_ok   else '🔴'} Groq API Key")
    st.markdown(f"{'🟢' if tavily_ok else '🔴'} Tavily Search Key")
    st.markdown("🟢 Polymarket API (public, no key needed)")


    st.divider()

    # ── Answer Mode ────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Answer Mode")
    mode = st.radio(
        label="",
        options=["concise", "detailed"],
        index=0 if st.session_state.answer_mode == "concise" else 1,
        format_func=lambda x: (
            "⚡ Concise — quick pick & top reasons"
            if x == "concise"
            else "🔬 Detailed — full evidence-based analysis"
        ),
    )
    st.session_state.answer_mode = mode

    st.divider()

    # ── Data Source Toggles ────────────────────────────────────────────────────
    st.markdown("### 🔌 Data Sources")
    use_poly = st.toggle("📈 Live Polymarket Data",  value=True)
    use_web  = st.toggle("🌐 Live Web / News Search", value=True)
    use_kb   = st.toggle("🗂️ Knowledge Base",         value=True)

    st.divider()

    # ── Knowledge Base Panel ───────────────────────────────────────────────────
    st.markdown("### 🗂️ Knowledge Base")

    # Default KB status
    dkb = st.session_state.default_vs
    ukb = st.session_state.user_vs

    col1, col2 = st.columns(2)
    col1.metric("Default KB", f"{dkb.total_chunks} chunks")
    col2.metric("Your KB",    f"{ukb.total_chunks} chunks")

    # Default KB sources list
    if dkb.total_chunks > 0:
        with st.expander(f"📚 Default KB — {len(dkb.unique_sources)} files"):
            for src in dkb.unique_sources:
                st.markdown(
                    f'<span class="kb-pill kb-pill-on">✓ {src}</span>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("⚠️ Default KB not built. Run `python knowledge_base/build_default_kb.py`")

    st.markdown("**Upload your own research files:**")
    uploaded_files = st.file_uploader(
        label="",
        type=["pdf", "txt", "md", "csv", "docx"],
        accept_multiple_files=True,
        help="PDF, TXT, MD, CSV, DOCX — chunked and added to your personal KB",
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("➕ Add to My KB", use_container_width=True, type="primary"):
            added, errors = 0, []
            for uf in uploaded_files:
                text, err = ingest_file(uf)
                if err:
                    errors.append(f"{uf.name}: {err}")
                elif text.strip():
                    try:
                        n = st.session_state.user_vs.add_documents(
                            [text], source_name=uf.name
                        )
                        added += n
                    except Exception as e:
                        errors.append(f"{uf.name}: {e}")
            if added:
                st.session_state.user_vs.save(VECTOR_STORE_PATH)
                # Invalidate cache so next load picks it up
                load_vector_stores.clear()
                st.success(f"✅ Added {added} chunks from {len(uploaded_files)} file(s)!")
            for err in errors:
                st.error(err)

    # User KB source list
    if ukb.total_chunks > 0:
        with st.expander(f"📁 Your KB — {len(ukb.unique_sources)} file(s)"):
            for src in ukb.unique_sources:
                st.markdown(
                    f'<span class="kb-pill kb-pill-on">✓ {src}</span>',
                    unsafe_allow_html=True,
                )
        if st.button("🗑️ Clear My KB", use_container_width=True, type="secondary"):
            st.session_state.user_vs = VectorStore()
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH)
            load_vector_stores.clear()
            st.success("Your KB cleared.")
            st.rerun()

    st.divider()

    # ── Recommended Resources ──────────────────────────────────────────────────
    with st.expander("💡 Recommended files to upload"):
        st.markdown("""
**Add these to get even better recommendations:**

| File | Source |
|------|--------|
| Polymarket resolution rules | polymarket.com/help |
| Historical election CSVs | Wikipedia, 538 |
| Sports team stats | sports-reference.com |
| Fed meeting transcripts | federalreserve.gov |
| Economic data (IMF/WB) | imf.org, worldbank.org |
| FDA approval history | fda.gov |
| Geopolitical risk reports | rand.org |
| Prediction market papers | SSRN / Google Scholar |
""")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        "<small style='color:#4b5563'>Gemini 2.5 Flash-Lite · Groq LLaMA 3.3 · "
        "HuggingFace MiniLM · Tavily · Polymarket API</small>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
prov_label = (
    '<span class="provider-gemini">Gemini 2.5 Flash-Lite</span>'
    if st.session_state.llm_provider == "gemini"
    else '<span class="provider-groq">Groq LLaMA 3.3 70B</span>'
)
st.markdown(
    f"## 📊 Polymarket Bet Optimizer &nbsp;·&nbsp; {prov_label}",
    unsafe_allow_html=True,
)
st.markdown(
    "Ask about **any Polymarket market** — I'll pull live odds, breaking news, "
    "and domain knowledge to give you a data-driven betting recommendation."
)

# ── KB status bar ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.markdown(
    f'<span class="kb-pill {"kb-pill-on" if st.session_state.default_vs.total_chunks > 0 else "kb-pill-off"}">'
    f'📚 Default KB: {st.session_state.default_vs.total_chunks} chunks</span>',
    unsafe_allow_html=True,
)
k2.markdown(
    f'<span class="kb-pill {"kb-pill-on" if st.session_state.user_vs.total_chunks > 0 else "kb-pill-off"}">'
    f'📁 Your KB: {st.session_state.user_vs.total_chunks} chunks</span>',
    unsafe_allow_html=True,
)
k3.markdown(
    f'<span class="kb-pill {"kb-pill-on" if use_poly else "kb-pill-off"}">📈 Polymarket: {"on" if use_poly else "off"}</span>',
    unsafe_allow_html=True,
)
k4.markdown(
    f'<span class="kb-pill {"kb-pill-on" if use_web else "kb-pill-off"}">🌐 Web Search: {"on" if use_web else "off"}</span>',
    unsafe_allow_html=True,
)

st.divider()

# ── Example prompts ────────────────────────────────────────────────────────────
with st.expander("💡 Example questions — click to use"):
    examples = [
        "Will the Fed cut rates in May 2026?",
        "Who will win the 2026 FIFA World Cup?",
        "Will Bitcoin reach $150K by end of 2026?",
        "Will there be a US recession in 2026?",
        "Will Apple release a new iPhone model in September 2026?",
        "Who will win the 2028 US Presidential Election?",
        "Will Ethereum ETF see positive inflows this week?",
        "Will the UK general election result in a Labour majority?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["prefill_query"] = ex

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def render_sources(msg: dict):
    """Render the source expander tabs for a message."""
    user_srcs    = msg.get("user_sources",    [])
    default_srcs = msg.get("default_sources", [])
    poly         = msg.get("poly_context",    "")
    web          = msg.get("web_context",     "")

    tab_labels = []
    if user_srcs:                       tab_labels.append("📁 Your KB")
    if default_srcs:                    tab_labels.append("📚 Default KB")
    if poly and poly.strip():           tab_labels.append("📈 Polymarket")
    if web  and web.strip():            tab_labels.append("🌐 Web Results")

    if not tab_labels:
        return

    with st.expander("📎 View sources & raw data"):
        tabs = st.tabs(tab_labels)
        ti = 0

        if user_srcs:
            with tabs[ti]:
                for chunk, source, score in user_srcs:
                    st.markdown(f"**`{source}`** &nbsp; score: `{score:.3f}`")
                    st.markdown(
                        f"<div class='source-box'>{chunk[:450]}{'…' if len(chunk)>450 else ''}</div>",
                        unsafe_allow_html=True,
                    )
            ti += 1

        if default_srcs:
            with tabs[ti]:
                for chunk, source, score in default_srcs:
                    st.markdown(f"**`{source}`** &nbsp; score: `{score:.3f}`")
                    st.markdown(
                        f"<div class='source-box'>{chunk[:450]}{'…' if len(chunk)>450 else ''}</div>",
                        unsafe_allow_html=True,
                    )
            ti += 1

        if poly and poly.strip():
            with tabs[ti]:
                st.markdown(poly)
            ti += 1

        if web and web.strip():
            with tabs[ti]:
                st.markdown(web)


for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="📊"):
            badge_cls = "badge-concise" if msg.get("mode") == "concise" else "badge-detailed"
            badge_lbl = "⚡ Concise" if msg.get("mode") == "concise" else "🔬 Detailed"
            st.markdown(
                f'<span class="badge {badge_cls}">{badge_lbl}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(msg["content"])
            render_sources(msg)

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT
# ═══════════════════════════════════════════════════════════════════════════════
prefill    = st.session_state.pop("prefill_query", "")
user_input = st.chat_input(
    "Ask about any Polymarket bet… e.g. 'Will the Fed cut rates in May 2026?'"
)
query = user_input or prefill

if query:
    # ── Pre-flight API key check ───────────────────────────────────────────────
    active_provider = st.session_state.llm_provider
    if active_provider == "gemini" and not gemini_ok:
        st.error(
            "❌ **Gemini API key not set.** Add your key to the .env file (GEMINI_API_KEY=...) and restart the app."
        )
        st.stop()
    if active_provider == "groq" and not groq_ok:
        st.error(
            "❌ **Groq API key not set.** Add your key to the .env file (GROQ_API_KEY=...) and restart the app."
        )
        st.stop()

    # ── Add user message ───────────────────────────────────────────────────────
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    # ── Generate response ──────────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="📊"):
        mode = st.session_state.answer_mode
        badge_cls = "badge-concise" if mode == "concise" else "badge-detailed"
        badge_lbl = "⚡ Concise" if mode == "concise" else "🔬 Detailed"
        st.markdown(
            f'<span class="badge {badge_cls}">{badge_lbl}</span>',
            unsafe_allow_html=True,
        )

        # Status indicator while pipeline runs
        steps = []
        if use_poly: steps.append("📈 Fetching live Polymarket odds…")
        if use_web:  steps.append("🌐 Searching latest news & web…")
        if use_kb:   steps.append("🗂️ Querying knowledge bases…")
        steps.append("🤖 Generating recommendation…")

        with st.status("🔍 Analysing…", expanded=False) as status:
            for s in steps:
                st.write(s)

            result = run_rag(
                user_query=query,
                user_vector_store=st.session_state.user_vs,
                default_vector_store=st.session_state.default_vs,
                answer_mode=mode,
                use_web_search=use_web,
                use_polymarket=use_poly,
                use_kb=use_kb,
            )
            status.update(label="✅ Analysis complete", state="complete", expanded=False)

        st.markdown(result["response"])

        # Save to chat history
        st.session_state.chat_history.append({
            "role":            "assistant",
            "content":         result["response"],
            "mode":            mode,
            "user_sources":    result.get("user_sources",    []),
            "default_sources": result.get("default_sources", []),
            "poly_context":    result.get("poly_context",    ""),
            "web_context":     result.get("web_context",     ""),
        })

        render_sources({
            "user_sources":    result.get("user_sources",    []),
            "default_sources": result.get("default_sources", []),
            "poly_context":    result.get("poly_context",    ""),
            "web_context":     result.get("web_context",     ""),
        })