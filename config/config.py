import os

# ── Load .env file automatically ─────────────────────────────────────────────
# Keys in .env are loaded into os.environ before anything else reads them.
# Install with: pip install python-dotenv  (already in requirements.txt)
try:
    from dotenv import load_dotenv
    load_dotenv()   # reads .env from the project root directory
except ImportError:
    pass   # dotenv not installed; fall back to plain env vars

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Options: "gemini" | "groq"
#
# Free tier comparison (as of March 2026):
#   gemini  → Gemini 2.5 Flash:  15 RPM, 1,500 req/day (stable, no date suffix)
#   groq    → LLaMA 3.3 70B:     ~30 RPM, ~500K tokens/day (very generous)
#
# Switch to "groq" if you hit Gemini daily limits.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")

# ── API Keys ──────────────────────────────────────────────────────────────────
# These are read from environment variables (set in .env or exported in shell).
# NEVER hard-code real keys here and NEVER commit .env to Git.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY",   "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY",  "")

# ── LLM Settings ──────────────────────────────────────────────────────────────
# gemini-2.5-flash is the correct stable model string (no date suffix).
LLM_MODEL       = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3

# Max output tokens — Gemini 2.5 Flash hard limit is 65,535.
# IMPORTANT: Gemini 2.5 Flash is a "thinking" model. Internal reasoning tokens
# count against this same budget. With the old value of 4,096, the model's own
# thinking consumed most of the budget and cut the visible response short.
# 16,384 gives plenty of room for both internal reasoning + full responses.
MAX_OUTPUT_TOKENS = 16384

# ── Embedding Settings ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, local, fast
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 80
TOP_K_RESULTS   = 5

# ── Tavily Web Search ─────────────────────────────────────────────────────────
TAVILY_MAX_RESULTS  = 6
TAVILY_SEARCH_DEPTH = "advanced"   # "basic" | "advanced"

# ── Polymarket API ────────────────────────────────────────────────────────────
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_BASE  = "https://clob.polymarket.com"

# ── Vector Store Paths ────────────────────────────────────────────────────────
VECTOR_STORE_PATH = "knowledge_base/vector_store"          # user uploads
DEFAULT_KB_PATH   = "knowledge_base/default_vector_store"  # pre-built KB

# ── Answer Modes ──────────────────────────────────────────────────────────────
ANSWER_MODES = {
    "concise": (
        "You are a sharp prediction-market analyst. "
        "Give a CONCISE recommendation in 3-5 bullet points max. "
        "Lead with the recommended option and your probability estimate. "
        "No long explanations — just the verdict and top reasons."
    ),
    "detailed": (
        "You are a deep-research prediction-market analyst. "
        "Provide a DETAILED analysis covering:\n"
        "1) Market context & current odds from Polymarket\n"
        "2) Key evidence from recent news & web data\n"
        "3) Base-rate / historical comparison\n"
        "4) Probability assessment with full reasoning\n"
        "5) Final recommendation with confidence level\n"
        "6) Key risks — reasons your recommendation could be wrong"
    ),
}