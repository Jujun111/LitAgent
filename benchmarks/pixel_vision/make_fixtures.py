from __future__ import annotations

from pathlib import Path


def ensure_fixtures(root: str | Path | None = None) -> list[Path]:
    base_dir = Path(root) if root else Path(__file__).resolve().parent
    output_dir = base_dir / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("reportlab is required to generate pixel vision benchmark fixtures.") from exc

    outputs = [
        output_dir / "vision_bars.pdf",
        output_dir / "vision_lines.pdf",
        output_dir / "vision_table_image.pdf",
    ]
    if all(path.exists() for path in outputs):
        return outputs

    width, height = letter

    c = canvas.Canvas(str(outputs[0]), pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Synthetic Ablation Chart")
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 92, "Figure 1. Ablation outcomes are shown for three conditions.")
    x0, y0 = 120, 190
    c.line(x0, y0, x0 + 340, y0)
    c.line(x0, y0, x0, y0 + 230)
    bars = [
        ("blue baseline", colors.blue, 205),
        ("orange ablation", colors.orange, 82),
        ("green recovery", colors.green, 148),
    ]
    for idx, (label, color, bar_height) in enumerate(bars):
        x = x0 + 55 + idx * 92
        c.setFillColor(color)
        c.rect(x, y0, 42, bar_height, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawCentredString(x + 21, y0 - 18, label)
    c.save()

    c = canvas.Canvas(str(outputs[1]), pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Synthetic Training Curves")
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 92, "Figure 2. Training dynamics are shown across five epochs.")
    x0, y0 = 105, 185
    c.line(x0, y0, x0 + 390, y0)
    c.line(x0, y0, x0, y0 + 245)
    red_points = [(1, 80), (2, 125), (3, 165), (4, 220), (5, 175)]
    blue_points = [(1, 215), (2, 180), (3, 140), (4, 100), (5, 65)]
    for points, color in ((red_points, colors.red), (blue_points, colors.blue)):
        c.setStrokeColor(color)
        scaled = [(x0 + epoch * 64, y0 + value) for epoch, value in points]
        for first, second in zip(scaled, scaled[1:]):
            c.line(first[0], first[1], second[0], second[1])
        c.setFillColor(color)
        for x, y in scaled:
            c.circle(x, y, 4, fill=1, stroke=0)
    c.setFillColor(colors.red)
    c.drawString(x0 + 405, y0 + 215, "red accuracy")
    c.setFillColor(colors.blue)
    c.drawString(x0 + 405, y0 + 190, "blue loss")
    c.save()

    c = canvas.Canvas(str(outputs[2]), pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Synthetic Result Table Image")
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 92, "Figure 3. Model comparison table rendered as an image.")
    x0, y0 = 115, 245
    col_w = [110, 90, 90]
    row_h = 34
    rows = [
        ("Model", "F1", "Latency"),
        ("Model A", "0.87", "24 ms"),
        ("Model B", "0.82", "18 ms"),
        ("Model C", "0.91", "31 ms"),
    ]
    for r, row in enumerate(rows):
        y = y0 + (len(rows) - r - 1) * row_h
        for col, text in enumerate(row):
            x = x0 + sum(col_w[:col])
            c.setFillColor(colors.lightgrey if r == 0 else colors.white)
            c.rect(x, y, col_w[col], row_h, fill=1, stroke=1)
            c.setFillColor(colors.black)
            c.drawString(x + 8, y + 12, text)
    c.save()

    return outputs


def main() -> int:
    paths = ensure_fixtures()
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
