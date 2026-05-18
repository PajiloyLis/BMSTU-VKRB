#!/usr/bin/env python3
"""
Разбиение текста на предложения по символу точки (.).
Точка в конце каждого фрагмента восстанавливается, если она была в исходном тексте;
хвост без финальной точки выводится как есть.
"""

import argparse
import sys


def split_by_dots(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []

    def flush(with_dot: bool) -> None:
        s = "".join(buf).strip()
        buf.clear()
        if not s:
            return
        out.append(s + "." if with_dot else s)

    for ch in text:
        if ch == ".":
            flush(with_dot=True)
        else:
            buf.append(ch)
    flush(with_dot=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Разбор текста на фрагменты по точкам (.)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Файл с текстом (UTF-8). Если не указан — читать stdin.",
    )
    args = parser.parse_args()

    if args.path:
        with open(args.path, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
        print(text)

    for sentence in split_by_dots(text):
        print(sentence)


if __name__ == "__main__":
    main()
