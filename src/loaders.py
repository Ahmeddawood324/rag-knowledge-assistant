"""Step 1 - Data integration.

Turns PDFs and Google Sheets into LangChain `Document` objects (plain text
+ metadata) so the rest of the pipeline does not care where data came from.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from pypdf import PdfReader

from .config import Settings


# --------------------------------------------------------------------------
# PDFs  (unstructured knowledge)
# --------------------------------------------------------------------------
def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdfs(pdf_dir: Path) -> list[Document]:
    """One Document per PDF page; keeps file name + page number for citations."""
    docs: list[Document] = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_no, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_no, "type": "pdf"},
                    )
                )
    return docs


# --------------------------------------------------------------------------
# Google Sheets  (structured knowledge)
# --------------------------------------------------------------------------
def google_sheet_csv_url(url: str) -> str:
    """Convert a normal Google Sheets link into its CSV export link.

    The sheet must be shared as "Anyone with the link can view".
    """
    sheet_id = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not sheet_id:
        raise ValueError(f"Not a Google Sheets URL: {url}")
    gid = re.search(r"[#&?]gid=(\d+)", url)
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id.group(1)}"
        f"/export?format=csv&gid={gid.group(1) if gid else 0}"
    )


def dataframe_to_documents(df: pd.DataFrame, name: str) -> list[Document]:
    """Convert a table into text.

    * one summary document describing the sheet (columns, number of rows)
    * one document per row, written as "Column: value | Column: value"
      so that an embedding model can match it against natural questions.
    """
    df = df.dropna(how="all").reset_index(drop=True)
    docs = [
        Document(
            page_content=(
                f"Sheet '{name}' has {len(df)} rows and these columns: "
                + ", ".join(map(str, df.columns))
                + "."
            ),
            metadata={"source": f"{name}.csv", "row": 0, "type": "sheet_summary"},
        )
    ]
    for idx, row in df.iterrows():
        parts = [
            f"{col}: {str(val).strip()}"
            for col, val in row.items()
            if pd.notna(val) and str(val).strip()
        ]
        docs.append(
            Document(
                page_content=f"Sheet {name} | " + " | ".join(parts),
                metadata={"source": f"{name}.csv", "row": int(idx) + 2, "type": "sheet"},
                # row number matches the spreadsheet (row 1 = header)
            )
        )
    return docs


def _read_table(location: str) -> pd.DataFrame:
    if location.startswith("http"):
        return pd.read_csv(google_sheet_csv_url(location))
    return pd.read_csv(location)


def load_sheets(settings: Settings) -> list[Document]:
    docs: list[Document] = []
    if settings.sheet_urls:
        for i, entry in enumerate(settings.sheet_urls, start=1):
            # accepted forms:  "https://docs.google.com/..."  or  "courses=https://..."
            named = re.match(r"^([\w-]+)=(https?://.+)$", entry)
            name, url = (named.group(1), named.group(2)) if named else (f"google_sheet_{i}", entry)
            try:
                docs += dataframe_to_documents(_read_table(url), name)
            except Exception as exc:  # network / permission problems
                warnings.warn(f"Could not load Google Sheet '{name}': {exc}")
    else:
        for csv_path in sorted(Path(settings.sheets_dir).glob("*.csv")):
            docs += dataframe_to_documents(pd.read_csv(csv_path), csv_path.stem)
    return docs


def load_all(settings: Settings) -> list[Document]:
    return load_pdfs(settings.pdf_dir) + load_sheets(settings)
