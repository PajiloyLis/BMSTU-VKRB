#!/usr/bin/env python3
from pathlib import Path
import argparse
import fitz  # PyMuPDF


PT_PER_MM = 72 / 25.4


def mm(value: float) -> float:
    return value * PT_PER_MM


def add_frame(
    page: fitz.Page,
    left_mm: float = 20,
    right_mm: float = 5,
    top_mm: float = 5,
    bottom_mm: float = 5,
    line_width: float = 0.7,
) -> None:
    page_rect = page.rect

    frame_rect = fitz.Rect(
        page_rect.x0 + mm(left_mm),
        page_rect.y0 + mm(top_mm),
        page_rect.x1 - mm(right_mm),
        page_rect.y1 - mm(bottom_mm),
    )

    page.draw_rect(
        frame_rect,
        color=(0, 0, 0),
        width=line_width,
        overlay=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Добавляет рамки к страницам PDF."
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="nir/release/report_kostr.pdf",
        help="Входной PDF-файл",
    )

    parser.add_argument(
        "output",
        nargs="?",
        default="nir/release/report-ramki.pdf",
        help="Выходной PDF-файл",
    )

    parser.add_argument("--left", type=float, default=30, help="Левое поле, мм")
    parser.add_argument("--right", type=float, default=15, help="Правое поле, мм")
    parser.add_argument("--top", type=float, default=20, help="Верхнее поле, мм")
    parser.add_argument("--bottom", type=float, default=20, help="Нижнее поле, мм")
    parser.add_argument("--width", type=float, default=0.7, help="Толщина линии")
    parser.add_argument(
        "--skip-first",
        action="store_true",
        help="Не добавлять рамку на первую страницу",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(input_path)

    for page_index, page in enumerate(doc):
        if args.skip_first and page_index == 0:
            continue

        add_frame(
            page,
            left_mm=args.left,
            right_mm=args.right,
            top_mm=args.top,
            bottom_mm=args.bottom,
            line_width=args.width,
        )

    doc.save(
        output_path,
        garbage=4,
        deflate=True,
        clean=True,
    )

    doc.close()

    print(f"Готово! Файл сохранен как: {output_path}")


if __name__ == "__main__":
    main()
