"""
utils/rag_pipeline.py
Orchestrates the full RAG pipeline:
  1. Search default KB (pre-built domain knowledge)
  2. Search user-uploaded KB
  3. Live Polymarket data fetch
  4. Live web/news search
  5. Prompt assembly
  6. LLM generation
"""

from models.embeddings import VectorStore, format_kb_context
from models.llm import build_prompt, get_llm_response
from utils.polymarket_fetcher import format_market_context
from utils.web_search import search_web, build_polymarket_search_query


def _search_kb(vs: VectorStore, query: str):
    """Search a VectorStore and return (hits, formatted_context)."""
    if vs.total_chunks == 0:
        return [], ""
    try:
        hits = vs.search(query)
        return hits, format_kb_context(hits)
    except Exception as e:
        return [], f"⚠️ KB retrieval error: {e}"


def run_rag(
    user_query: str,
    user_vector_store: VectorStore,
    default_vector_store: VectorStore,
    answer_mode: str = "detailed",
    use_web_search: bool = True,
    use_polymarket: bool = True,
    use_kb: bool = True,
) -> dict:
    """
    Full RAG pipeline.

    Returns a dict with keys:
        response         – final LLM answer
        kb_context       – combined KB context (default + user)
        web_context      – live web search results
        poly_context     – live Polymarket data
        user_sources     – hits from user KB
        default_sources  – hits from default KB
    """
    result = {
        "response":        "",
        "kb_context":      "",
        "web_context":     "",
        "poly_context":    "",
        "user_sources":    [],
        "default_sources": [],
    }

    # ── 1. Knowledge Base retrieval ──────────────────────────────────────────
    if use_kb:
        # Default KB (pre-built domain knowledge)
        default_hits, default_ctx = _search_kb(default_vector_store, user_query)
        result["default_sources"] = default_hits

        # User-uploaded KB
        user_hits, user_ctx = _search_kb(user_vector_store, user_query)
        result["user_sources"] = user_hits

        # Merge — user uploads first (more specific), then default domain KB
        parts = []
        if user_ctx:
            parts.append(f"### User-Uploaded Research\n{user_ctx}")
        if default_ctx:
            parts.append(f"### Domain Knowledge Base\n{default_ctx}")
        result["kb_context"] = "\n\n".join(parts)

    # ── 2. Polymarket live data ──────────────────────────────────────────────
    if use_polymarket:
        try:
            result["poly_context"] = format_market_context(user_query)
        except Exception as e:
            result["poly_context"] = f"⚠️ Polymarket fetch error: {e}"

    # ── 3. Live web / news search ────────────────────────────────────────────
    if use_web_search:
        try:
            enhanced_query = build_polymarket_search_query(user_query)
            result["web_context"] = search_web(enhanced_query)
        except Exception as e:
            result["web_context"] = f"⚠️ Web search error: {e}"

    # ── 4. Build prompt + call LLM ───────────────────────────────────────────
    try:
        prompt = build_prompt(
            user_query=user_query,
            kb_context=result["kb_context"],
            web_context=result["web_context"],
            polymarket_context=result["poly_context"],
            answer_mode=answer_mode,
        )
        result["response"] = get_llm_response(prompt)
    except Exception as e:
        result["response"] = f"❌ Pipeline error: {e}"

    return result