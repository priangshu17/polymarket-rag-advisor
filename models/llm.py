"""
models/llm.py
LLM interface — Google Gemini 2.5 Flash-Lite (free tier, 1000 RPD)
Uses the NEW Google GenAI SDK: `from google import genai`
Groq (LLaMA 3.3 70B) available as fallback — 500K tokens/day free.
"""

from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.config import (
    GEMINI_API_KEY,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    ANSWER_MODES,
)

# ── Singleton Gemini client ────────────────────────────────────────────────────
_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    """Return (or lazily create) the Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(
    user_query: str,
    kb_context: str,
    web_context: str,
    polymarket_context: str,
    answer_mode: str = "detailed",
) -> str:
    """Assemble the full RAG prompt."""
    sections = []

    if polymarket_context.strip():
        sections.append(f"## Live Polymarket Data\n{polymarket_context}")

    if web_context.strip():
        sections.append(f"## Live Web / News Evidence\n{web_context}")

    if kb_context.strip():
        sections.append(f"## Knowledge Base Context\n{kb_context}")

    context_block = (
        "\n\n".join(sections) if sections else "No additional context available."
    )

    system_persona = ANSWER_MODES.get(answer_mode, ANSWER_MODES["detailed"])

    prompt = f"""{system_persona}

---
{context_block}
---

User Question: {user_query}

Using ALL the context above, provide your prediction-market analysis and betting recommendation.
Always state:
1. The recommended option to bet on
2. Your estimated probability (e.g. 65%)
3. Key supporting reasons
4. Key risks / reasons you could be wrong

If you are uncertain about any aspect, say so clearly.
"""
    return prompt


# ── Gemini response ────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """Call Gemini using the new Google GenAI SDK."""
    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        return response.text

    except APIError as e:
        # Surface quota / rate-limit errors clearly
        if "429" in str(e) or "quota" in str(e).lower():
            return (
                "⚠️ **Gemini rate limit reached.** You've hit the free tier quota "
                "(1,000 req/day for Flash-Lite). Switch to Groq in config.py "
                "(`LLM_PROVIDER = 'groq'`) or try again tomorrow.\n\n"
                f"Raw error: {e}"
            )
        return f"❌ Gemini API Error: {e}"

    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"


# ── Groq fallback ──────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    """Call Groq (LLaMA 3.3 70B) — 500K tokens/day free."""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return chat_completion.choices[0].message.content

    except ImportError:
        return "❌ Groq not installed. Run: pip install groq"
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
            return f"⚠️ Groq rate limit hit: {e}"
        return f"❌ Groq Error: {str(e)}"


# ── Public entry point ─────────────────────────────────────────────────────────

def get_llm_response(prompt: str) -> str:
    """
    Route to the configured LLM provider.
    LLM_PROVIDER = 'gemini'  →  Gemini 2.5 Flash-Lite (1,000 RPD free)
    LLM_PROVIDER = 'groq'    →  Groq LLaMA 3.3 70B   (500K tokens/day free)
    """
    provider = LLM_PROVIDER.lower().strip()

    if provider == "groq":
        return _call_groq(prompt)
    else:
        return _call_gemini(prompt)