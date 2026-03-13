# 📊 Polymarket RAG Advisor

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that helps you make smarter bets on **any Polymarket market** — politics, crypto, sports, economics, science, and more.

It combines live Polymarket odds, real-time web/news search, and a curated knowledge base to give you data-driven betting recommendations with probability estimates.

---

## 🧠 How It Works

```
Your question
     │
     ├──► 📈 Live Polymarket API   → current odds, volume, liquidity
     ├──► 🌐 Tavily Web Search     → latest news & analysis
     ├──► 📚 Default Knowledge Base → domain expertise (pre-built)
     └──► 📁 Your Uploaded Files   → your own research
                    │
                    ▼
          🤖 Gemini 2.5 Flash / Groq LLaMA 3.3 70B
                    │
                    ▼
       Recommendation + Probability Estimate
```

---

## ⚙️ Tech Stack

| Layer          | Technology                       | Notes                     |
| -------------- | -------------------------------- | ------------------------- |
| LLM (primary)  | Google Gemini 2.5 Flash          | 1,500 req/day free        |
| LLM (fallback) | Groq LLaMA 3.3 70B               | ~500K tokens/day free     |
| Embeddings     | HuggingFace `all-MiniLM-L6-v2` | Free, runs fully locally  |
| Vector store   | FAISS                            | Free, runs fully locally  |
| Web search     | Tavily                           | 1,000 searches/month free |
| Market data    | Polymarket public API            | No key required           |
| UI             | Streamlit                        |                           |
| Env management | python-dotenv                    |                           |

---

## 📁 Project Structure

```
polymarket_rag/
├── .env.example                        ← Copy to .env and fill in your keys
├── .gitignore                          ← Ensures .env is never committed
├── app.py                              ← Main Streamlit UI
├── requirements.txt                    ← All dependencies
│
├── config/
│   └── config.py                       ← All settings; reads keys from .env
│
├── models/
│   ├── llm.py                          ← Gemini 2.5 Flash + Groq LLaMA 3.3 70B
│   └── embeddings.py                   ← HuggingFace MiniLM + FAISS vector store
│
├── utils/
│   ├── rag_pipeline.py                 ← Orchestrates the full RAG flow
│   ├── polymarket_fetcher.py           ← Live Polymarket market data & odds
│   ├── web_search.py                   ← Tavily live news/web search
│   └── file_ingestion.py               ← PDF, TXT, DOCX, CSV parser
│
└── knowledge_base/
    ├── build_default_kb.py             ← Run once to build default vector store
    ├── sources/                        ← Pre-built domain knowledge (.txt files)
    │   ├── 01_prediction_market_fundamentals.txt
    │   ├── 02_political_elections_guide.txt
    │   ├── 03_crypto_financial_markets.txt
    │   ├── 04_sports_markets_guide.txt
    │   └── 05_science_tech_geopolitical.txt
    ├── default_vector_store/           ← Auto-generated on first run (FAISS index)
    └── vector_store/                   ← Auto-generated when you upload files
```

---

## 🚀 Setup & Installation

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd polymarket_rag
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Get your free API keys

You need **one LLM key** (Gemini or Groq) and the Tavily key.

| Key                | Free Tier                | Sign Up                                          |
| ------------------ | ------------------------ | ------------------------------------------------ |
| `GEMINI_API_KEY` | 1,500 req/day, 15 RPM    | [aistudio.google.com](https://aistudio.google.com/) |
| `GROQ_API_KEY`   | ~500K tokens/day, 30 RPM | [console.groq.com](https://console.groq.com/)       |
| `TAVILY_API_KEY` | 1,000 searches/month     | [tavily.com](https://tavily.com/)                   |

### Step 4 — Configure your `.env` file

```bash
# Copy the template
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
LLM_PROVIDER=gemini

GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

> ⚠️ **Never commit `.env` to Git.** It is already listed in `.gitignore`.
> The `.env.example` file (with placeholder values) is safe to commit.

### Step 5 — Build the default knowledge base

This only needs to be run  **once** . It embeds the 5 pre-built domain knowledge files into a local FAISS vector store.

```bash
python knowledge_base/build_default_kb.py
```

> The app will also auto-build this on first launch if you skip this step.

### Step 6 — Run the app

```bash
streamlit run app.py
```

---

## 🔑 API Key Security

API keys are managed **exclusively** through the `.env` file. They are never stored in the code and cannot be entered or viewed through the Streamlit UI.

| File                 | Committed to Git? | Purpose                                          |
| -------------------- | :---------------: | ------------------------------------------------ |
| `.env`             |     ❌ Never     | Your real API keys (gitignored)                  |
| `.env.example`     |      ✅ Yes      | Template with placeholder values                 |
| `config/config.py` |      ✅ Yes      | Reads keys from env vars — no hardcoded secrets |

If a key is missing when you submit a query, the app will display:

> *"Add your key to the .env file (KEY_NAME=...) and restart the app."*

---

## 🗂️ Knowledge Base

The app ships with two independent knowledge bases.

### Default KB (pre-built, ships with the project)

Five domain knowledge files covering the most common Polymarket categories:

| File                                      | Contents                                                                      |
| ----------------------------------------- | ----------------------------------------------------------------------------- |
| `01_prediction_market_fundamentals.txt` | How prediction markets work, calibration, Kelly criterion, common biases      |
| `02_political_elections_guide.txt`      | Election base rates, polling interpretation, legislative & regulatory markets |
| `03_crypto_financial_markets.txt`       | Fed rate decisions, Bitcoin/ETH signals, CME FedWatch, economic indicators    |
| `04_sports_markets_guide.txt`           | NFL/NBA/MLB/soccer win rates, home advantage stats, injury signal timing      |
| `05_science_tech_geopolitical.txt`      | FDA approval rates, AI capability markets, geopolitical risk signals          |

The default KB loads automatically on startup. If the FAISS index doesn't exist yet, it is built from the source `.txt` files on first launch.

### Your KB (user uploads, persisted across sessions)

Upload your own research files via the sidebar. Supported formats:

| Format    | Notes                       |
| --------- | --------------------------- |
| `.pdf`  | Text extracted page by page |
| `.txt`  | Plain text                  |
| `.md`   | Markdown                    |
| `.csv`  | Rows joined as text         |
| `.docx` | Word documents              |

Uploaded files are chunked, embedded, and saved to `knowledge_base/vector_store/` — they persist across restarts. You can clear your KB at any time from the sidebar.

### Recommended files to upload for better results

| Resource                          | Where to get it               |
| --------------------------------- | ----------------------------- |
| Polymarket resolution criteria    | polymarket.com/help           |
| Historical election results (CSV) | Wikipedia, FiveThirtyEight    |
| Sports team stats (CSV)           | sports-reference.com          |
| Fed meeting transcripts           | federalreserve.gov            |
| Economic indicator data           | imf.org, worldbank.org        |
| FDA drug approval history         | fda.gov                       |
| Geopolitical risk reports         | rand.org, atlanticcouncil.org |
| Prediction market research papers | SSRN, Google Scholar          |

---

## 🤖 LLM Providers

Switch providers any time using the sidebar radio button — no restart needed.

### Gemini 2.5 Flash (default)

* Model string: `gemini-2.5-flash` (stable — no date suffix)
* Free tier:  **1,500 requests/day** , 15 RPM
* Max output: 65,535 tokens (configured to 16,384 — see note below)
* Note: this is a **thinking model** — internal reasoning tokens count against the output budget. The limit is set to 16,384 to give enough headroom for both reasoning and full responses.

### Groq LLaMA 3.3 70B (fallback)

* Model string: `llama-3.3-70b-versatile`
* Free tier:  **~500,000 tokens/day** , ~30 RPM
* Recommended for heavy testing and development where Gemini daily limits are tight

---

## 💬 Features

| Feature                            | Description                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| **Any Polymarket topic**     | Politics, crypto, sports, economics, science, entertainment — no topic restrictions |
| **⚡ Concise mode**          | 3–5 bullet recommendation with probability estimate and top reasons                 |
| **🔬 Detailed mode**         | Full analysis: market context, evidence, base rates, probability reasoning, risks    |
| **📚 Default KB**            | Ships with 5 pre-built domain knowledge files, auto-loads on startup                 |
| **📁 User file upload**      | Add your own PDFs, CSVs, research docs — persisted across sessions                  |
| **📈 Live Polymarket data**  | Real-time market odds, 24h volume, liquidity via public API                          |
| **🌐 Live web search**       | Latest news via Tavily, 6 sources per query, advanced search depth                   |
| **🔌 Source toggles**        | Independently enable/disable Polymarket data, web search, and KB                     |
| **📎 Source transparency**   | View exactly which KB chunks, market data, and news informed each answer             |
| **🔑 Secure key management** | Keys loaded from `.env`only — no UI inputs, never logged                          |

---

## 🛠️ Configuration Reference

All settings are in `config/config.py`. The most useful ones to tweak:

| Setting                 | Default                | Description                                  |
| ----------------------- | ---------------------- | -------------------------------------------- |
| `LLM_PROVIDER`        | `"gemini"`           | `"gemini"`or `"groq"`                    |
| `LLM_MODEL`           | `"gemini-2.5-flash"` | Gemini model string                          |
| `MAX_OUTPUT_TOKENS`   | `16384`              | Max tokens in LLM response                   |
| `LLM_TEMPERATURE`     | `0.3`                | Lower = more focused, higher = more creative |
| `TOP_K_RESULTS`       | `5`                  | Number of KB chunks retrieved per query      |
| `CHUNK_SIZE`          | `600`                | Words per KB chunk                           |
| `TAVILY_MAX_RESULTS`  | `6`                  | Web search results per query                 |
| `TAVILY_SEARCH_DEPTH` | `"advanced"`         | `"basic"`or `"advanced"`                 |

---

## ❓ Troubleshooting

**Responses cut off mid-way**
Gemini 2.5 Flash is a thinking model — its internal reasoning tokens count against `MAX_OUTPUT_TOKENS`. If responses still cut off, increase `MAX_OUTPUT_TOKENS` in `config.py` (max is 65,535).

**404 NOT_FOUND from Gemini API**
The model name has a date suffix or typo. The correct stable string is exactly `gemini-2.5-flash` — no preview suffix, no date.

**Default KB not loading**
Run `python knowledge_base/build_default_kb.py` manually, or simply start the app — it will auto-build on first launch if the source files exist in `knowledge_base/sources/`.

**API key not detected**
Ensure `.env` is in the project root directory (same folder as `app.py`), that `python-dotenv` is installed (`pip install python-dotenv`), and that the key name matches exactly (e.g. `GEMINI_API_KEY`, not `GEMINI_KEY`).

**Polymarket data not showing**
The Polymarket API is public and requires no key. If it fails, it's usually a temporary outage — the app will still respond using web search and KB context.
