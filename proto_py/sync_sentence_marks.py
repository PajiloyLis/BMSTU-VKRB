#!/usr/bin/env python3
"""
Перечитывает файл sentences: строки, которые разбираются в IP (CYK), помечает
префиксом «- »; у остальных префикс убирается. Текст для разбора — строка без
ведущего «-» и пробелов после него.
"""

from pathlib import Path

from cyk import build_cyk_table
from grammar import binarize_grammar, fix_grammar, load_grammar
from morph_utils import tokenize_input

DEFAULT_PATH = Path(__file__).resolve().parent / "sentences"


def line_parses(line: str, gram: dict) -> bool:
    s = line.strip()
    if s.startswith("-"):
        s = s[1:].lstrip()
    if not s:
        return False
    tokens = tokenize_input(s)
    n = len(tokens)
    if n == 0:
        return False
    dp = build_cyk_table(tokens, gram)
    return "IP" in dp[0][n]


def sync_file(path: Path) -> None:
    gram = load_grammar(path.parent / "grammar.json")
    gram = fix_grammar(gram)
    gram = binarize_grammar(gram)

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for raw in raw_lines:
        if not raw.strip():
            out.append(raw)
            continue
        ok = line_parses(raw, gram)
        body = raw.strip()
        if body.startswith("-"):
            body = body[1:].lstrip()
        out.append("- " + body if ok else body)

    text = "\n".join(out)
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    sync_file(DEFAULT_PATH)


if __name__ == "__main__":
    main()
