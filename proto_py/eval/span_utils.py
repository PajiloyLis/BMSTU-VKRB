"""Span extraction and scoring for RuConst evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from grammar import BINARIZATION_AUX_PREFIX

Span = tuple[str, int, int]  # (label, start, end) — end exclusive, 0-based

DEFAULT_LABELS = frozenset({"NP", "VP", "PP"})


def is_binarization_aux(tag: str) -> bool:
    return isinstance(tag, str) and tag.startswith(BINARIZATION_AUX_PREFIX)


def is_punct_token(token: dict[str, Any]) -> bool:
    tagsets = token.get("tagsets") or []
    if not tagsets:
        return False
    pos = tagsets[0][0] if tagsets[0] else ""
    return pos == "PUNCT"


def gold_content_tokens(tokens: list[dict[str, Any]], include_punct: bool = False) -> list[str]:
    """Gold word forms used for alignment (optionally excluding punctuation)."""
    result: list[str] = []
    for t in tokens:
        if not include_punct and is_punct_token(t):
            continue
        result.append(t["token"])
    return result


def itoken_to_index(itoken: int, tokens: list[dict[str, Any]], include_punct: bool) -> int | None:
    """Map 1-based gold itoken to 0-based index in content-token coordinates."""
    idx = 0
    for t in tokens:
        if not include_punct and is_punct_token(t):
            continue
        if t["itoken"] == itoken:
            return idx
        idx += 1
    return None


def gold_spans(
    constituents: list[dict[str, Any]],
    tokens: list[dict[str, Any]],
    labels: Iterable[str] = DEFAULT_LABELS,
    include_punct: bool = False,
) -> set[Span]:
    label_set = set(labels)
    spans: set[Span] = set()
    for c in constituents:
        name = c.get("name")
        if name not in label_set:
            continue
        raw_tokens = c.get("tokens") or []
        if not raw_tokens:
            continue
        indices: list[int] = []
        for itoken, _word in raw_tokens:
            pos = itoken_to_index(itoken, tokens, include_punct)
            if pos is None:
                continue
            indices.append(pos)
        if not indices:
            continue
        start, end = min(indices), max(indices) + 1
        spans.add((name, start, end))
    return spans


def _leaf_words_in_order(node: dict[str, Any]) -> list[str]:
    if "word" in node:
        return [node["word"]]
    if is_binarization_aux(node.get("tag", "")):
        words: list[str] = []
        for child in node.get("children", []):
            words.extend(_leaf_words_in_order(child))
        return words
    words = []
    for child in node.get("children", []):
        words.extend(_leaf_words_in_order(child))
    return words


def align_leaves_to_indices(leaves: list[str], content_tokens: list[str]) -> list[int] | None:
    """Greedy left-to-right alignment of leaf words to content token indices."""
    if not leaves:
        return []
    indices: list[int] = []
    cursor = 0
    for word in leaves:
        while cursor < len(content_tokens) and content_tokens[cursor] != word:
            cursor += 1
        if cursor >= len(content_tokens):
            return None
        indices.append(cursor)
        cursor += 1
    return indices


def pred_spans(
    tree: dict[str, Any],
    content_tokens: list[str],
    labels: Iterable[str] = DEFAULT_LABELS,
) -> set[Span]:
    label_set = set(labels)
    spans: set[Span] = set()

    def walk(node: dict[str, Any]) -> None:
        tag = node.get("tag", "")
        if is_binarization_aux(tag):
            for child in node.get("children", []):
                walk(child)
            return

        if tag in label_set:
            leaves = _leaf_words_in_order(node)
            indices = align_leaves_to_indices(leaves, content_tokens)
            if indices is not None:
                spans.add((tag, min(indices), max(indices) + 1))

        for child in node.get("children", []):
            walk(child)

    walk(tree)
    return spans


def tokens_match(gold_tokens: list[str], parser_tokens: list[str]) -> bool:
    return gold_tokens == parser_tokens


@dataclass(frozen=True)
class SpanScores:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def exact_match(self) -> bool:
        return self.fp == 0 and self.fn == 0


def score_spans(gold: set[Span], pred: set[Span]) -> SpanScores:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return SpanScores(tp=tp, fp=fp, fn=fn)


def format_spans(spans: set[Span]) -> str:
    parts = sorted(f"{label}[{start}:{end}]" for label, start, end in spans)
    return ";".join(parts)


def pick_best_tree(
    trees: list[dict[str, Any]],
    gold: set[Span],
    content_tokens: list[str],
    labels: Iterable[str] = DEFAULT_LABELS,
) -> tuple[int | None, set[Span], SpanScores]:
    """Return (best_index_1based, best_pred_spans, best_scores)."""
    if not trees:
        empty = SpanScores(tp=0, fp=0, fn=len(gold))
        return None, set(), empty

    best_idx: int | None = None
    best_pred: set[Span] = set()
    best_scores = SpanScores(tp=-1, fp=0, fn=0)
    best_f1 = -1.0

    for i, tree in enumerate(trees, start=1):
        pred = pred_spans(tree, content_tokens, labels)
        scores = score_spans(gold, pred)
        f1 = scores.f1
        if f1 > best_f1 or (f1 == best_f1 and scores.tp > best_scores.tp):
            best_f1 = f1
            best_idx = i
            best_pred = pred
            best_scores = scores

    return best_idx, best_pred, best_scores
