"""
knowledge_base/build_default_kb.py

Run this ONCE to embed all source .txt files into the default FAISS vector store.
After running, the app will automatically load these embeddings.

Usage:
    cd polymarket_rag
    python knowledge_base/build_default_kb.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.embeddings import VectorStore
from config.config import DEFAULT_KB_PATH

SOURCES_DIR = os.path.join(os.path.dirname(__file__), "sources")


def build():
    print("🔧 Building default knowledge base...")
    vs = VectorStore()
    total = 0

    if not os.path.isdir(SOURCES_DIR):
        print(f"❌ Sources directory not found: {SOURCES_DIR}")
        sys.exit(1)

    txt_files = sorted(
        f for f in os.listdir(SOURCES_DIR) if f.endswith(".txt")
    )
    if not txt_files:
        print("⚠️  No .txt files found in sources/")
        return

    for fname in txt_files:
        fpath = os.path.join(SOURCES_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            n = vs.add_documents([text], source_name=fname)
            print(f"  ✅ {fname}: {n} chunks")
            total += n
        except Exception as e:
            print(f"  ❌ {fname}: {e}")

    vs.save(DEFAULT_KB_PATH)
    print(f"\n✅ Default KB built — {total} total chunks saved to: {DEFAULT_KB_PATH}")


if __name__ == "__main__":
    build()