"""Интеграционные тесты CYK с реальной грамматикой и морфологией."""

from pathlib import Path

import pytest

from cyk import build_cyk_table
from grammar import binarize_grammar, fix_grammar, invert_grammar, load_grammar
from morph_utils import preprocess_tokens, tokenize_input

GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammar.json"


def _has_symbol(dp, i, j, sym):
    return bool(dp[i][j].get(sym))


@pytest.fixture(scope="module")
def grammar_indexes():
    grammar = binarize_grammar(fix_grammar(load_grammar(str(GRAMMAR_PATH))))
    return invert_grammar(grammar)


@pytest.mark.parametrize(
    "text",
    [
        "Кот спит.",
        "Я хочу пить.",
        "Где магазин?",
    ],
)
def test_real_grammar_parses_short_sentences(text, grammar_indexes):
    unary_index, binary_index = grammar_indexes
    tokens = tokenize_input(text)
    pairs = preprocess_tokens(tokens)
    n = len(tokens)
    dp = build_cyk_table(pairs, unary_index, binary_index)
    assert _has_symbol(dp, 0, n, "IP"), f"Нет IP для: {text!r}, tokens={tokens}"


def test_real_grammar_builds_table_for_mismatched_number(grammar_indexes):
    """Несогласованное число: таблица строится без ошибок (КЭ-6 на стабах)."""
    unary_index, binary_index = grammar_indexes
    pairs = [
        ("кот", [{"pos": "N", "number": "sg", "case": "nomn", "gender": "m"}]),
        ("спят", [{"pos": "V", "number": "pl", "person": "3", "tense": "pres", "verb_form": "fin"}]),
    ]
    dp = build_cyk_table(pairs, unary_index, binary_index)
    assert dp is not None
    assert _has_symbol(dp, 0, 1, "NP")
    assert _has_symbol(dp, 1, 2, "VP")
