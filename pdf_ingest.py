from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


CAPTION_PATTERN = re.compile(r"^\s*((?:Figure|Fig\.|Table)\s+\d+[:.]\s+.+)$", re.IGNORECASE)


def configure_docling_environment() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HOME", str(Path(".hf-cache").resolve()))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "paper"


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def provenance_page(item: Any, fallback: int = 1) -> int:
    prov = getattr(item, "prov", None) or []
    if prov:
        return int(getattr(prov[0], "page_no", fallback) or fallback)
    return fallback


def split_page_markdown(markdown: str, page_no: int, paper_id: str, source_path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    blocks = [compact_text(block) for block in re.split(r"\n\s*\n", markdown) if compact_text(block)]
    for block_index, block in enumerate(blocks, start=1):
        if block == "<!-- image -->" or block.startswith("|"):
            continue
        if CAPTION_PATTERN.match(block):
            continue
        chunks.append(
            {
                "text": block,
                "page_number": page_no,
                "source_ref": f"{source_path.name}#page={page_no}:text-{block_index}",
                "chunk_type": "text",
                "paper_id": paper_id,
            }
        )
    return chunks


def extract_caption_chunks(markdown: str, page_no: int, paper_id: str, source_path: Path) -> list[dict[str, Any]]:
    captions: list[dict[str, Any]] = []
    for caption_index, line in enumerate(markdown.splitlines(), start=1):
        match = CAPTION_PATTERN.match(line)
        if not match:
            continue
        captions.append(
            {
                "text": compact_text(match.group(1)),
                "page_number": page_no,
                "source_ref": f"{source_path.name}#page={page_no}:caption-{caption_index}",
                "chunk_type": "caption",
                "paper_id": paper_id,
            }
        )
    return captions


def parse_pdf_to_paper_record(pdf_path: str | Path, paper_id: str | None = None) -> dict[str, Any]:
    configure_docling_environment()
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError("Docling is required for PDF ingestion. Install requirements-pdf.txt first.") from exc

    source_path = Path(pdf_path).resolve()
    resolved_paper_id = paper_id or slugify(source_path.stem)
    result = DocumentConverter().convert(source_path)
    document = result.document

    pages = sorted((getattr(document, "pages", {}) or {}).keys()) or [1]
    page_markdown = {page_no: document.export_to_markdown(page_no=page_no) for page_no in pages}
    full_text_chunks: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    for page_no, markdown in page_markdown.items():
        full_text_chunks.extend(split_page_markdown(markdown, page_no, resolved_paper_id, source_path))
        captions.extend(extract_caption_chunks(markdown, page_no, resolved_paper_id, source_path))

    tables_markdown: list[dict[str, Any]] = []
    for table_index, table in enumerate(getattr(document, "tables", []) or [], start=1):
        page_no = provenance_page(table)
        tables_markdown.append(
            {
                "text": table.export_to_markdown(doc=document),
                "page_number": page_no,
                "source_ref": f"{source_path.name}#page={page_no}:table-{table_index}",
                "chunk_type": "table",
                "paper_id": resolved_paper_id,
            }
        )

    figures: list[dict[str, Any]] = []
    for figure_index, picture in enumerate(getattr(document, "pictures", []) or [], start=1):
        page_no = provenance_page(picture)
        figures.append(
            {
                "caption": compact_text(getattr(picture, "caption_text", "") or ""),
                "page_number": page_no,
                "source_ref": f"{source_path.name}#page={page_no}:figure-{figure_index}",
                "chunk_type": "figure",
                "paper_id": resolved_paper_id,
            }
        )

    title = source_path.stem.replace("_", " ").replace("-", " ").strip().title()
    for chunk in full_text_chunks:
        text = chunk.get("text", "")
        if text.startswith("#"):
            title = text.lstrip("# ").strip() or title
            break

    return {
        "paper_id": resolved_paper_id,
        "title": title,
        "abstract": "",
        "year": None,
        "venue": "PDF layout fixture",
        "citation_count": None,
        "reference_count": None,
        "influential_citation_count": None,
        "authors": [],
        "fields_of_study": [],
        "url": None,
        "open_access_pdf_url": str(source_path),
        "external_ids": {},
        "full_text_chunks": full_text_chunks,
        "tables_markdown": tables_markdown,
        "figures": figures,
        "captions": captions,
        "docling_markdown": "\n\n".join(page_markdown.values()),
    }


def discover_pdfs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.pdf"))
    return [path]


def parse_pdf_path(path: str | Path) -> list[dict[str, Any]]:
    return [parse_pdf_to_paper_record(pdf_path) for pdf_path in discover_pdfs(Path(path))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse full-paper PDFs into LitAgent fixed paper records.")
    parser.add_argument("path", type=Path, help="PDF file or directory of PDFs.")
    parser.add_argument("--output", type=Path, default=Path("evidence/pdf_ingest_output.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = parse_pdf_path(args.path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "papers": len(records),
                "text_chunks": sum(len(item.get("full_text_chunks", [])) for item in records),
                "table_chunks": sum(len(item.get("tables_markdown", [])) for item in records),
                "caption_chunks": sum(len(item.get("captions", [])) for item in records),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
