#!/usr/bin/env python3
"""Evaluate CYK parser against RuConst parsed_ruconst corpus."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.span_utils import (
    DEFAULT_LABELS,
    SpanScores,
    format_spans,
    gold_content_tokens,
    gold_spans,
    pick_best_tree,
    tokens_match,
)
from parser_service import ParserEngine

DEFAULT_CORPUS = Path("/home/ivan/Study/dicklom/parsed_ruconst")
DEFAULT_GRAMMAR = Path(__file__).parent / "grammar.json"


class ParseTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise ParseTimeout()


@dataclass
class SentenceResult:
    id: str
    text: str
    source: str
    gold_token_count: int
    parser_token_count: int
    parsed: bool
    tree_count: int
    trees_evaluated: int
    best_tree_index: int | None
    gold_spans_str: str
    pred_spans_str: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    exact_match: bool
    status: str
    elapsed_ms: float


@dataclass
class EvalSummary:
    total: int = 0
    ok: int = 0
    unparsed: int = 0
    token_mismatch: int = 0
    timeout: int = 0
    exact_match_count: int = 0
    micro_tp: int = 0
    micro_fp: int = 0
    micro_fn: int = 0
    macro_precision_sum: float = 0.0
    macro_recall_sum: float = 0.0
    macro_f1_sum: float = 0.0
    scored_sentences: int = 0
    by_source: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, row: SentenceResult) -> None:
        self.total += 1
        if row.status == "ok":
            self.ok += 1
            self.scored_sentences += 1
            self.micro_tp += row.tp
            self.micro_fp += row.fp
            self.micro_fn += row.fn
            self.macro_precision_sum += row.precision
            self.macro_recall_sum += row.recall
            self.macro_f1_sum += row.f1
            if row.exact_match:
                self.exact_match_count += 1
        elif row.status == "unparsed":
            self.unparsed += 1
            gold_count = row.fn
            self.micro_fn += gold_count
        elif row.status == "token_mismatch":
            self.token_mismatch += 1
            self.micro_fn += row.fn
        elif row.status == "timeout":
            self.timeout += 1
            self.micro_fn += row.fn

        src = row.source or "unknown"
        bucket = self.by_source.setdefault(
            src,
            {"total": 0, "ok": 0, "exact_match": 0, "micro_tp": 0, "micro_fp": 0, "micro_fn": 0},
        )
        bucket["total"] += 1
        if row.status == "ok":
            bucket["ok"] += 1
            bucket["micro_tp"] += row.tp
            bucket["micro_fp"] += row.fp
            bucket["micro_fn"] += row.fn
            if row.exact_match:
                bucket["exact_match"] += 1

    def _prf(self, tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    def to_dict(self) -> dict[str, Any]:
        micro_p, micro_r, micro_f1 = self._prf(self.micro_tp, self.micro_fp, self.micro_fn)
        n = self.scored_sentences or 1
        by_source_out = {}
        for src, b in self.by_source.items():
            sp, sr, sf1 = self._prf(b["micro_tp"], b["micro_fp"], b["micro_fn"])
            by_source_out[src] = {
                **b,
                "micro_precision": sp,
                "micro_recall": sr,
                "micro_f1": sf1,
                "exact_match_rate": b["exact_match"] / b["total"] if b["total"] else 0.0,
            }
        return {
            "total": self.total,
            "ok": self.ok,
            "unparsed": self.unparsed,
            "token_mismatch": self.token_mismatch,
            "timeout": self.timeout,
            "exact_match_count": self.exact_match_count,
            "corpus_exact_match_rate": self.exact_match_count / self.total if self.total else 0.0,
            "micro": {
                "tp": self.micro_tp,
                "fp": self.micro_fp,
                "fn": self.micro_fn,
                "precision": micro_p,
                "recall": micro_r,
                "f1": micro_f1,
            },
            "macro": {
                "precision": self.macro_precision_sum / n if self.scored_sentences else 0.0,
                "recall": self.macro_recall_sum / n if self.scored_sentences else 0.0,
                "f1": self.macro_f1_sum / n if self.scored_sentences else 0.0,
            },
            "by_source": by_source_out,
        }


def evaluate_sentence(
    data: dict[str, Any],
    engine: ParserEngine,
    *,
    max_trees: int,
    include_punct: bool,
    timeout_sec: float | None,
    labels: frozenset[str],
) -> SentenceResult:
    sent_id = str(data.get("id", ""))
    text = data.get("text", "")
    source = str(data.get("source", ""))
    tokens = data.get("tokens") or []
    constituents = data.get("constituents") or []

    gold_content = gold_content_tokens(tokens, include_punct=include_punct)
    gold = gold_spans(constituents, tokens, labels=labels, include_punct=include_punct)
    empty_scores = SpanScores(tp=0, fp=0, fn=len(gold))

    start = time.perf_counter()
    old_handler = None
    if timeout_sec and timeout_sec > 0:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)

    try:
        parse_result = engine.parse_sentence_all(text, max_trees=max_trees)
    except ParseTimeout:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SentenceResult(
            id=sent_id,
            text=text,
            source=source,
            gold_token_count=len(gold_content),
            parser_token_count=0,
            parsed=False,
            tree_count=0,
            trees_evaluated=0,
            best_tree_index=None,
            gold_spans_str=format_spans(gold),
            pred_spans_str="",
            tp=0,
            fp=0,
            fn=len(gold),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            exact_match=False,
            status="timeout",
            elapsed_ms=elapsed_ms,
        )
    finally:
        if timeout_sec and timeout_sec > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    elapsed_ms = (time.perf_counter() - start) * 1000
    parser_tokens = parse_result.tokens
    if not include_punct:
        parser_content = [t for t in parser_tokens if t not in {".", ",", ";", ":"}]
    else:
        parser_content = parser_tokens

    if not tokens_match(gold_content, parser_content):
        return SentenceResult(
            id=sent_id,
            text=text,
            source=source,
            gold_token_count=len(gold_content),
            parser_token_count=len(parser_content),
            parsed=parse_result.parsed,
            tree_count=parse_result.tree_count,
            trees_evaluated=len(parse_result.trees),
            best_tree_index=None,
            gold_spans_str=format_spans(gold),
            pred_spans_str="",
            tp=0,
            fp=0,
            fn=len(gold),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            exact_match=False,
            status="token_mismatch",
            elapsed_ms=elapsed_ms,
        )

    if not parse_result.parsed:
        return SentenceResult(
            id=sent_id,
            text=text,
            source=source,
            gold_token_count=len(gold_content),
            parser_token_count=len(parser_content),
            parsed=False,
            tree_count=0,
            trees_evaluated=0,
            best_tree_index=None,
            gold_spans_str=format_spans(gold),
            pred_spans_str="",
            tp=0,
            fp=0,
            fn=len(gold),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            exact_match=False,
            status="unparsed",
            elapsed_ms=elapsed_ms,
        )

    tree_dicts = [t.tree for t in parse_result.trees]
    best_idx, best_pred, scores = pick_best_tree(tree_dicts, gold, gold_content, labels=labels)

    return SentenceResult(
        id=sent_id,
        text=text,
        source=source,
        gold_token_count=len(gold_content),
        parser_token_count=len(parser_content),
        parsed=True,
        tree_count=parse_result.tree_count,
        trees_evaluated=len(parse_result.trees),
        best_tree_index=best_idx,
        gold_spans_str=format_spans(gold),
        pred_spans_str=format_spans(best_pred),
        tp=scores.tp,
        fp=scores.fp,
        fn=scores.fn,
        precision=scores.precision,
        recall=scores.recall,
        f1=scores.f1,
        exact_match=scores.exact_match,
        status="ok",
        elapsed_ms=elapsed_ms,
    )


CSV_FIELDS = [
    "id",
    "text",
    "source",
    "gold_token_count",
    "parser_token_count",
    "parsed",
    "tree_count",
    "trees_evaluated",
    "best_tree_index",
    "gold_spans",
    "pred_spans",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "exact_match",
    "status",
    "elapsed_ms",
]


def row_to_csv_dict(row: SentenceResult) -> dict[str, Any]:
    return {
        "id": row.id,
        "text": row.text,
        "source": row.source,
        "gold_token_count": row.gold_token_count,
        "parser_token_count": row.parser_token_count,
        "parsed": row.parsed,
        "tree_count": row.tree_count,
        "trees_evaluated": row.trees_evaluated,
        "best_tree_index": row.best_tree_index if row.best_tree_index is not None else "",
        "gold_spans": row.gold_spans_str,
        "pred_spans": row.pred_spans_str,
        "tp": row.tp,
        "fp": row.fp,
        "fn": row.fn,
        "precision": f"{row.precision:.6f}",
        "recall": f"{row.recall:.6f}",
        "f1": f"{row.f1:.6f}",
        "exact_match": int(row.exact_match),
        "status": row.status,
        "elapsed_ms": f"{row.elapsed_ms:.2f}",
    }


def iter_corpus_files(corpus_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(corpus_dir.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    return files


def run_evaluation(args: argparse.Namespace) -> EvalSummary:
    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    engine = ParserEngine(Path(args.grammar))
    labels = frozenset(args.labels.split(",")) if args.labels else DEFAULT_LABELS
    files = iter_corpus_files(corpus_dir, args.limit)
    summary = EvalSummary()
    rows: list[SentenceResult] = []

    try:
        from tqdm import tqdm
        iterator: Any = tqdm(files, desc="Evaluating")
    except ImportError:
        iterator = files

    for path in iterator:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        row = evaluate_sentence(
            data,
            engine,
            max_trees=args.max_trees,
            include_punct=args.include_punct,
            timeout_sec=args.timeout_per_sentence,
            labels=labels,
        )
        rows.append(row)
        summary.add(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv_dict(row))

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate parser on RuConst parsed_ruconst corpus")
    parser.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS))
    parser.add_argument("--grammar", type=str, default=str(DEFAULT_GRAMMAR))
    parser.add_argument("--output", type=str, default="ruconst_eval.csv")
    parser.add_argument("--summary", type=str, default="ruconst_eval_summary.json")
    parser.add_argument("--max-trees", type=int, default=500, help="Max parser trees per sentence (oracle pool)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N JSON files")
    parser.add_argument("--include-punct", action="store_true", help="Include punctuation in token alignment and spans")
    parser.add_argument(
        "--timeout-per-sentence",
        type=float,
        default=None,
        help="Timeout in seconds per sentence (Unix only)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="NP,VP,PP",
        help="Comma-separated constituent labels to compare",
    )
    args = parser.parse_args()

    summary = run_evaluation(args)
    stats = summary.to_dict()
    print("\n=== Evaluation summary ===")
    print(f"Total:           {stats['total']}")
    print(f"OK:              {stats['ok']}")
    print(f"Unparsed:        {stats['unparsed']}")
    print(f"Token mismatch:  {stats['token_mismatch']}")
    print(f"Timeout:         {stats['timeout']}")
    print(f"Exact match:     {stats['exact_match_count']} ({stats['corpus_exact_match_rate']:.2%})")
    micro = stats["micro"]
    print(f"Micro P/R/F1:    {micro['precision']:.4f} / {micro['recall']:.4f} / {micro['f1']:.4f}")
    macro = stats["macro"]
    print(f"Macro P/R/F1:    {macro['precision']:.4f} / {macro['recall']:.4f} / {macro['f1']:.4f}")
    print(f"CSV:             {args.output}")
    print(f"Summary JSON:    {args.summary}")


if __name__ == "__main__":
    main()
