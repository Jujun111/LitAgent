from __future__ import annotations

from pathlib import Path

from pdf_ingest import parse_pdf_path


FIXTURE_DIR = Path("benchmarks/full_paper_layout/fixtures")


def main() -> int:
    records = parse_pdf_path(FIXTURE_DIR)
    assert len(records) >= 3
    for record in records:
        assert record["paper_id"]
        assert record["full_text_chunks"]
        assert record["tables_markdown"]
        assert record["captions"]
        assert any(chunk.get("page_number") for chunk in record["full_text_chunks"])
        assert any(chunk.get("source_ref") for chunk in record["tables_markdown"])
        assert any(chunk.get("source_ref") for chunk in record["captions"])

    print(
        "PDF ingestion smoke test passed: "
        f"{len(records)} PDFs, "
        f"{sum(len(item['full_text_chunks']) for item in records)} text chunks, "
        f"{sum(len(item['tables_markdown']) for item in records)} table chunks, "
        f"{sum(len(item['captions']) for item in records)} caption chunks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
