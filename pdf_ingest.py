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


def item_caption_text(item: Any) -> str:
    caption = getattr(item, "caption_text", "")
    if callable(caption):
        return ""
    return compact_text(str(caption or ""))


def provenance_page(item: Any, fallback: int = 1) -> int:
    prov = getattr(item, "prov", None) or []
    if prov:
        return int(getattr(prov[0], "page_no", fallback) or fallback)
    return fallback


def provenance_bbox(item: Any) -> dict[str, Any] | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    bbox = getattr(prov[0], "bbox", None)
    if bbox is None:
        return None
    return {
        "l": float(getattr(bbox, "l", 0.0)),
        "t": float(getattr(bbox, "t", 0.0)),
        "r": float(getattr(bbox, "r", 0.0)),
        "b": float(getattr(bbox, "b", 0.0)),
        "coord_origin": str(getattr(getattr(bbox, "coord_origin", ""), "value", getattr(bbox, "coord_origin", ""))),
    }


def crop_box_from_bbox(
    bbox: dict[str, Any] | None,
    page_width: float,
    page_height: float,
    scale: float,
    padding_px: int,
) -> tuple[int, int, int, int]:
    image_width = int(page_width * scale)
    image_height = int(page_height * scale)
    if not bbox:
        return (0, 0, image_width, image_height)

    left = float(bbox["l"]) * scale
    right = float(bbox["r"]) * scale
    origin = str(bbox.get("coord_origin", "")).upper()
    if "BOTTOMLEFT" in origin:
        top = (page_height - float(bbox["t"])) * scale
        bottom = (page_height - float(bbox["b"])) * scale
    else:
        top = float(bbox["t"]) * scale
        bottom = float(bbox["b"]) * scale

    x0 = max(0, int(min(left, right)) - padding_px)
    x1 = min(image_width, int(max(left, right)) + padding_px)
    y0 = max(0, int(min(top, bottom)) - padding_px)
    y1 = min(image_height, int(max(top, bottom)) + padding_px)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, image_width, image_height)
    return (x0, y0, x1, y1)


def render_pdf_crop(
    pdf_path: Path,
    page_no: int,
    output_path: Path,
    bbox: dict[str, Any] | None = None,
    scale: float = 2.0,
    padding_px: int = 24,
) -> None:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is required for PDF image crop export.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_no - 1]
    page_width, page_height = page.get_size()
    image = page.render(scale=scale).to_pil()
    crop_box = crop_box_from_bbox(bbox, page_width, page_height, scale, padding_px)
    image.crop(crop_box).save(output_path)
    if hasattr(page, "close"):
        page.close()
    if hasattr(pdf, "close"):
        pdf.close()


def export_visual_assets(
    document: Any,
    source_path: Path,
    paper_id: str,
    output_dir: Path,
    include_page_images: bool = False,
) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    def add_asset(item: Any, visual_type: str, index: int) -> None:
        page_no = provenance_page(item)
        bbox = provenance_bbox(item)
        filename = f"{slugify(source_path.stem)}-page-{page_no}-{visual_type}-{index}.png"
        image_path = output_dir / filename
        render_pdf_crop(source_path, page_no, image_path, bbox=bbox)
        assets.append(
            {
                "paper_id": paper_id,
                "visual_type": visual_type,
                "image_path": str(image_path.resolve()),
                "page_number": page_no,
                "source_ref": f"{source_path.name}#page={page_no}:{visual_type}-{index}",
                "bbox": bbox,
                "caption": item_caption_text(item),
                "chunk_type": "visual_asset",
            }
        )

    for index, picture in enumerate(getattr(document, "pictures", []) or [], start=1):
        add_asset(picture, "figure", index)

    for index, table in enumerate(getattr(document, "tables", []) or [], start=1):
        add_asset(table, "table", index)

    if include_page_images:
        pages = sorted((getattr(document, "pages", {}) or {}).keys()) or [1]
        for page_no in pages:
            filename = f"{slugify(source_path.stem)}-page-{page_no}-page-1.png"
            image_path = output_dir / filename
            render_pdf_crop(source_path, page_no, image_path, bbox=None)
            assets.append(
                {
                    "paper_id": paper_id,
                    "visual_type": "page",
                    "image_path": str(image_path.resolve()),
                    "page_number": page_no,
                    "source_ref": f"{source_path.name}#page={page_no}:page-image",
                    "bbox": None,
                    "caption": "",
                    "chunk_type": "visual_asset",
                }
            )

    return assets


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


def parse_pdf_to_paper_record(
    pdf_path: str | Path,
    paper_id: str | None = None,
    export_visuals: bool = False,
    visual_output_dir: str | Path | None = None,
    include_page_images: bool = False,
) -> dict[str, Any]:
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
                "bbox": provenance_bbox(table),
            }
        )

    figures: list[dict[str, Any]] = []
    for figure_index, picture in enumerate(getattr(document, "pictures", []) or [], start=1):
        page_no = provenance_page(picture)
        figures.append(
            {
                "caption": item_caption_text(picture),
                "page_number": page_no,
                "source_ref": f"{source_path.name}#page={page_no}:figure-{figure_index}",
                "chunk_type": "figure",
                "paper_id": resolved_paper_id,
                "bbox": provenance_bbox(picture),
            }
        )

    visual_assets: list[dict[str, Any]] = []
    if export_visuals:
        output_dir = Path(visual_output_dir) if visual_output_dir else Path("evidence/visual_crops") / resolved_paper_id
        visual_assets = export_visual_assets(
            document,
            source_path=source_path,
            paper_id=resolved_paper_id,
            output_dir=output_dir,
            include_page_images=include_page_images,
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
        "visual_assets": visual_assets,
        "docling_markdown": "\n\n".join(page_markdown.values()),
    }


def discover_pdfs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.pdf"))
    return [path]


def parse_pdf_path(
    path: str | Path,
    export_visuals: bool = False,
    visual_output_dir: str | Path | None = None,
    include_page_images: bool = False,
) -> list[dict[str, Any]]:
    return [
        parse_pdf_to_paper_record(
            pdf_path,
            export_visuals=export_visuals,
            visual_output_dir=Path(visual_output_dir) / slugify(pdf_path.stem) if visual_output_dir else None,
            include_page_images=include_page_images,
        )
        for pdf_path in discover_pdfs(Path(path))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse full-paper PDFs into LitAgent fixed paper records.")
    parser.add_argument("path", type=Path, help="PDF file or directory of PDFs.")
    parser.add_argument("--output", type=Path, default=Path("evidence/pdf_ingest_output.json"))
    parser.add_argument("--export-visuals", action="store_true", help="Render figure/table crops as PNG files.")
    parser.add_argument("--include-page-images", action="store_true", help="Also render full page images.")
    parser.add_argument("--visual-output-dir", type=Path, default=Path("evidence/visual_crops"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = parse_pdf_path(
        args.path,
        export_visuals=args.export_visuals,
        visual_output_dir=args.visual_output_dir,
        include_page_images=args.include_page_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "papers": len(records),
                "text_chunks": sum(len(item.get("full_text_chunks", [])) for item in records),
                "table_chunks": sum(len(item.get("tables_markdown", [])) for item in records),
                "caption_chunks": sum(len(item.get("captions", [])) for item in records),
                "visual_assets": sum(len(item.get("visual_assets", [])) for item in records),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
