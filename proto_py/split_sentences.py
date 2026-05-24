#!/usr/bin/env python3
"""
Разбиение текста на предложения по символам конца предложения.
Исходная пунктуация сохраняется, пробельные символы внутри предложения
нормализуются до одного пробела.
"""

import argparse
import re
import sys


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        s = re.sub(r"\s+", " ", "".join(buf)).strip()
        buf.clear()
        if not s:
            return
        out.append(s)

    for ch in text:
        buf.append(ch)
        if ch in ".?!":
            flush()
    flush()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Разбор текста на предложения по . ? !."
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

    for sentence in split_sentences(text):
        print(sentence)


if __name__ == "__main__":
    main()
