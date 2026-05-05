from __future__ import annotations

from pathlib import Path

from benchmarks.pixel_vision.make_fixtures import ensure_fixtures
from pdf_ingest import parse_pdf_path
from vision_ingest import extract_vision_observations, vision_observations_to_chunks


FIXTURE_DIR = Path("benchmarks/pixel_vision")


def main() -> int:
    ensure_fixtures(FIXTURE_DIR)
    records = parse_pdf_path(
        FIXTURE_DIR / "generated",
        export_visuals=True,
        visual_output_dir=Path("evidence/visual_crops/pixel_smoke"),
        include_page_images=True,
    )
    assert len(records) >= 3
    assert sum(len(record.get("visual_assets", [])) for record in records) >= 3

    for record in records:
        for asset in record.get("visual_assets", []):
            asset["expected_visual_facts"] = [
                f"Pixel-level visual evidence was extracted from {asset['source_ref']}."
            ]

    vision_result = extract_vision_observations(
        query="pixel-level figure and table understanding",
        papers=records,
        provider="mock",
    )
    assert vision_result["vision_asset_count"] >= 3
    assert vision_result["vision_observations"]
    assert vision_observations_to_chunks(vision_result["vision_observations"])

    print(
        "Pixel vision smoke test passed: "
        f"{len(records)} PDFs, "
        f"{vision_result['vision_asset_count']} visual assets, "
        f"{len(vision_result['vision_observations'])} mock observations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
