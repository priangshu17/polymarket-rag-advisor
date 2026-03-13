"""
utils/polymarket_fetcher.py
Fetch live markets, prices, and trade data from Polymarket's public APIs.
No API key required — all public endpoints.
"""

import re
import requests
from typing import Optional
from config.config import POLYMARKET_GAMMA_BASE, POLYMARKET_CLOB_BASE


HEADERS = {"User-Agent": "PolymarketRAG/1.0"}
TIMEOUT = 10


def search_markets(query: str, limit: int = 5) -> list[dict]:
    """Search Polymarket markets by keyword."""
    try:
        params = {
            "q": query,
            "limit": limit,
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
        }
        r = requests.get(
            f"{POLYMARKET_GAMMA_BASE}/markets",
            params=params,
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        # API may return a list directly or wrap in a key
        if isinstance(data, list):
            return data[:limit]
        return data.get("markets", data.get("results", []))[:limit]
    except Exception as e:
        return [{"error": str(e)}]


def get_market_detail(condition_id: str) -> dict:
    """Get full detail for a single market by its condition ID."""
    try:
        r = requests.get(
            f"{POLYMARKET_GAMMA_BASE}/markets/{condition_id}",
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_market_prices(condition_id: str) -> dict:
    """Fetch current YES/NO token prices from the CLOB."""
    try:
        r = requests.get(
            f"{POLYMARKET_CLOB_BASE}/markets/{condition_id}",
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _price_to_prob(price: float | str) -> str:
    """Convert a CLOB price (0-1) to a readable probability string."""
    try:
        p = float(price)
        return f"{p * 100:.1f}%"
    except Exception:
        return str(price)


def format_market_context(query: str) -> str:
    """
    Main entry point called by the RAG pipeline.
    Searches for relevant markets and formats them into a readable context block.
    """
    markets = search_markets(query, limit=5)

    if not markets or (len(markets) == 1 and "error" in markets[0]):
        err = markets[0].get("error", "unknown") if markets else "no results"
        return f"⚠️ Could not fetch Polymarket data: {err}"

    lines = ["### Relevant Polymarket Markets\n"]
    for mkt in markets:
        if "error" in mkt:
            continue
        title      = mkt.get("question") or mkt.get("title") or "Unknown market"
        volume     = mkt.get("volume", mkt.get("volume24hr", "N/A"))
        liquidity  = mkt.get("liquidity", "N/A")
        end_date   = mkt.get("endDate", mkt.get("end_date_iso", "N/A"))
        outcomes   = mkt.get("outcomes", [])         # list of strings
        prices     = mkt.get("outcomePrices", [])    # list of price strings

        lines.append(f"**Market:** {title}")
        lines.append(f"  • Ends: {end_date}")
        if volume != "N/A":
            lines.append(f"  • 24h Volume: ${float(volume):,.0f}" if _is_numeric(volume) else f"  • Volume: {volume}")
        if liquidity != "N/A":
            lines.append(f"  • Liquidity: ${float(liquidity):,.0f}" if _is_numeric(liquidity) else f"  • Liquidity: {liquidity}")

        if outcomes and prices and len(outcomes) == len(prices):
            lines.append("  • Current Odds:")
            for outcome, price in zip(outcomes, prices):
                lines.append(f"      – {outcome}: {_price_to_prob(price)}")
        elif outcomes:
            lines.append(f"  • Outcomes: {', '.join(outcomes)}")

        lines.append("")   # blank line between markets

    return "\n".join(lines)


def _is_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False