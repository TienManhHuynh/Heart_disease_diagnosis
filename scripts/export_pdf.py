"""Xuất reports/final_report.md thành PDF. Cần extra [report] hoặc bundled Python."""

import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Keep the exported report Unicode-capable on Windows, Linux and macOS.
    candidates = (
        (Path("C:/Windows/Fonts"), ("arial.ttf", "arialbd.ttf", "ariali.ttf")),
        (
            Path("/usr/share/fonts/truetype/dejavu"),
            ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"),
        ),
        (Path("/Library/Fonts"), ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf")),
    )
    selected = next(
        (
            (directory, names)
            for directory, names in candidates
            if all((directory / name).exists() for name in names)
        ),
        None,
    )
    if selected is None:
        raise FileNotFoundError("Install Arial or DejaVu Sans before exporting the Vietnamese PDF.")
    font_dir, filenames = selected
    for name, filename in zip(("Arial", "Arial-Bold", "Arial-Italic"), filenames):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        "Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold"
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyVI",
            fontName="Arial",
            fontSize=9.6,
            leading=14,
            spaceAfter=7,
            textColor=colors.HexColor("#233447"),
            splitLongWords=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleVI",
            parent=styles["BodyVI"],
            fontName="Arial-Bold",
            fontSize=28,
            leading=33,
            spaceBefore=16,
            spaceAfter=16,
            textColor=colors.HexColor("#123d58"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingVI",
            parent=styles["BodyVI"],
            fontName="Arial-Bold",
            fontSize=15,
            leading=20,
            spaceBefore=12,
            spaceAfter=10,
            keepWithNext=True,
            textColor=colors.HexColor("#126f82"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableVI", parent=styles["BodyVI"], fontSize=7.5, leading=9.5, spaceAfter=0
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaptionVI",
            parent=styles["BodyVI"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#607285"),
            spaceAfter=12,
        )
    )
    available = A4[0] - 84

    def inline(text):
        text = escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        return re.sub(r"`([^`]+)`", r'<font color="#126f82">\1</font>', text)

    source = ROOT / "reports/final_report.md"
    lines = source.read_text(encoding="utf-8").splitlines()
    story, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["TitleVI"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["HeadingVI"]))
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            path = source.parent / match[2]
            if path.name == "bar.png":
                story.append(
                    Paragraph(
                        "Biểu đồ bar toàn cục được đính kèm trong thư mục shap; bảng mean abs SHAP ở trên trình bày cùng đại lượng.",
                        styles["CaptionVI"],
                    )
                )
                i += 1
                continue
            with PILImage.open(path) as image:
                width, height = image.size
            factor = min(available / width, 250 / height)
            story.append(
                Image(str(path), width=width * factor, height=height * factor, hAlign="CENTER")
            )
            story.append(Spacer(1, 5))
            story.append(Paragraph(inline(match[1]), styles["CaptionVI"]))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                fields = [field.strip() for field in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", field or " ") for field in fields):
                    rows.append(fields)
                i += 1
            count = len(rows[0])
            if any(len(row) != count for row in rows):
                raise ValueError(f"Invalid table near line {i}")
            if count == 3 and rows[0][0] == "Biến thể":
                widths = [75, 42, available - 117]
            elif count == 7:
                widths = [96, 63, 78, 58, 58, 58, available - 411]
            elif count == 2:
                widths = [120, available - 120]
            else:
                widths = [available / count] * count
            data = [[Paragraph(inline(value), styles["TableVI"]) for value in row] for row in rows]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceef1")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f3f7fa")],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#9ac5ce")),
                    ]
                )
            )
            block = [table, Spacer(1, 12)]
            story.extend([KeepTogether(block)] if len(rows) <= 10 else block)
            continue
        else:
            paragraphs = [line]
            while (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].startswith(("#", "|", "![", "<!--"))
            ):
                i += 1
                paragraphs.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(paragraphs)), styles["BodyVI"]))
        i += 1

    def decorate(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d4e2e8"))
        canvas.line(42, 39, A4[0] - 42, 39)
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.HexColor("#607285"))
        canvas.drawString(42, 26, "AI Project 3.1 | Cleveland Heart Disease | Báo cáo thực nghiệm")
        canvas.drawRightString(A4[0] - 42, 26, f"Trang {doc.page}")

    destination = ROOT / "output/pdf/Heart_Disease_Project_Report.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=40,
        bottomMargin=53,
        title="Heart Disease Diagnosis - Project 3.1",
        author="Project 3.1",
    )
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(destination)


if __name__ == "__main__":
    main()
