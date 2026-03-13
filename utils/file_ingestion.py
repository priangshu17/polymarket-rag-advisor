"""
utils/file_ingestion.py
Parse uploaded files (PDF, TXT, DOCX, CSV, MD) into plain text.
"""

import io
from typing import Tuple


def ingest_file(uploaded_file) -> Tuple[str, str]:
    """
    Accept a Streamlit UploadedFile object.
    Returns (extracted_text, error_message).
    error_message is empty string on success.
    """
    filename = uploaded_file.name.lower()

    try:
        # ── Plain text / Markdown ─────────────────────────────────────────────
        if filename.endswith((".txt", ".md")):
            return uploaded_file.read().decode("utf-8", errors="ignore"), ""

        # ── CSV ───────────────────────────────────────────────────────────────
        elif filename.endswith(".csv"):
            import csv
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            reader  = csv.reader(io.StringIO(content))
            rows    = [", ".join(row) for row in reader]
            return "\n".join(rows), ""

        # ── PDF ───────────────────────────────────────────────────────────────
        elif filename.endswith(".pdf"):
            try:
                import pdfplumber
                text_pages = []
                with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_pages.append(page_text)
                return "\n\n".join(text_pages), ""
            except ImportError:
                return "", "pdfplumber not installed. Run: pip install pdfplumber"

        # ── DOCX ──────────────────────────────────────────────────────────────
        elif filename.endswith(".docx"):
            try:
                from docx import Document
                doc   = Document(io.BytesIO(uploaded_file.read()))
                paras = [p.text for p in doc.paragraphs if p.text.strip()]
                return "\n\n".join(paras), ""
            except ImportError:
                return "", "python-docx not installed. Run: pip install python-docx"

        else:
            return "", f"Unsupported file type: {uploaded_file.name}"

    except Exception as e:
        return "", f"Error reading file '{uploaded_file.name}': {str(e)}"