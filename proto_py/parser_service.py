from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cyk import build_cyk_table
from grammar import binarize_grammar, fix_grammar, load_grammar
from morph_utils import tokenize_input
from tree_utils import extract_trees, tree_to_bracket


@dataclass(frozen=True)
class ParsedTree:
    index: int
    bracket: str
    tree: dict[str, Any]


@dataclass(frozen=True)
class ParseResult:
    text: str
    tokens: list[str]
    parsed: bool
    root_symbol: str
    tree_count: int
    trees: list[ParsedTree]
    diagnostics: dict[str, int]


class ParserEngine:
    def __init__(self, grammar_path: Path, root_symbol: str = "IP") -> None:
        grammar = load_grammar(str(grammar_path))
        self.grammar = binarize_grammar(fix_grammar(grammar))
        self.unary_index, self.binary_index = invert_grammar(self.grammar)
        self.root_symbol = root_symbol

    def parse_sentence(self, text: str, max_trees: int = 5) -> ParseResult:
        normalized = text.strip()
        tokens = tokenize_input(normalized)
        token_feature_pairs = preprocess_tokens(tokens)
        dp = build_cyk_table(token_feature_pairs, self.unary_index, self.binary_index)
        top_cell = dp[0][len(tokens)]
        parsed = self.root_symbol in top_cell
        trees: list[ParsedTree] = []
        tree_count = 0

        if parsed:
            all_trees = extract_trees(0, len(tokens), self.root_symbol, tokens, dp, self.grammar, {})
            tree_count = len(all_trees)
            for index, tree in enumerate(all_trees[:max_trees], start=1):
                trees.append(ParsedTree(index=index, bracket=tree_to_bracket(tree), tree=tree))

        return ParseResult(
            text=normalized,
            tokens=tokens,
            parsed=parsed,
            root_symbol=self.root_symbol,
            tree_count=tree_count,
            trees=trees,
            diagnostics={symbol: len(features) for symbol, features in top_cell.items()},
        )

