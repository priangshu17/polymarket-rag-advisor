"""
utils/web_search.py
Live web + news search using Tavily API.
"""

from tavily import TavilyClient
from config.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS, TAVILY_SEARCH_DEPTH


def search_web(query: str, max_results: int = TAVILY_MAX_RESULTS) -> str:
    """
    Run a Tavily search and return a formatted context string.
    Falls back gracefully on error.
    """
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth=TAVILY_SEARCH_DEPTH,
            max_results=max_results,
            include_answer=True,          # Tavily's own synthesis snippet
            include_raw_content=False,
        )

        lines = ["### Live Web Search Results\n"]

        # Tavily's synthesized answer (if available)
        tavily_answer = response.get("answer", "")
        if tavily_answer:
            lines.append(f"**Summary:** {tavily_answer}\n")

        results = response.get("results", [])
        for i, r in enumerate(results, 1):
            title   = r.get("title", "No title")
            url     = r.get("url", "")
            content = r.get("content", "").strip()
            score   = r.get("score", 0)

            lines.append(f"**[{i}] {title}**")
            if url:
                lines.append(f"  Source: {url}")
            if content:
                # Trim very long snippets
                snippet = content[:500] + ("…" if len(content) > 500 else "")
                lines.append(f"  {snippet}")
            lines.append("")

        return "\n".join(lines) if len(lines) > 1 else "No web results found."

    except Exception as e:
        return f"⚠️ Web search error: {str(e)}"


def build_polymarket_search_query(user_query: str) -> str:
    """
    Augment the user query with Polymarket-relevant terms
    to get better news & analysis results.
    """
    boosters = "prediction market odds news latest 2025"
    return f"{user_query} {boosters}"