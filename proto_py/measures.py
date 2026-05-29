#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import statistics
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Предполагаем, что модули вашего парсера находятся в текущей директории
from morph_utils import tokenize_input, preprocess_tokens
from cyk import build_cyk_table
from grammar import binarize_grammar, fix_grammar, load_grammar, invert_grammar
from tree_utils import extract_trees  # это extract_trees из tree_utils (одноимённая функция)


def generate_sentence(length: int) -> str:
    """
    Генерирует простое русское предложение заданной длины (количество слов).
    Для чётного length: чередование существительное + глагол, между парами вставляется 'и'.
    Для нечётного length: к чётному предложению добавляется наречие в конец.
    """
    nouns = ["мальчик", "девочка", "кот", "собака", "мама", "папа"]
    verbs = ["спит", "играет", "бегает", "ест", "пьёт"]
    adverbs = ["быстро", "медленно", "громко", "тихо", "хорошо"]

    if length == 1:
        return nouns[0]

    # Базовое чётное предложение
    pairs = length // 2
    parts = []
    for i in range(pairs):
        parts.append(nouns[i % len(nouns)])
        parts.append(verbs[i % len(verbs)])
        if i < pairs - 1:
            parts.append("и")
    sentence = " ".join(parts)

    # Для нечётной длины добавим наречие после последнего глагола
    if length % 2 == 1:
        sentence += " " + adverbs[0]

    return sentence


def measure_parser(text: str, parser_engine, root_symbol="IP") -> tuple[float, float]:
    """
    Выполняет полный разбор текста и возвращает (time_cyk, time_extract) в секундах.
    Использует методы parser_engine (с уже загруженной грамматикой и индексами).
    """
    # Токенизация и предобработка признаков
    tokens = tokenize_input(text)
    token_feature_pairs = preprocess_tokens(tokens)

    # Замер времени build_cyk_table
    start = time.perf_counter()
    dp = build_cyk_table(
        token_feature_pairs,
        parser_engine.unary_index,
        parser_engine.binary_index
    )
    time_cyk = time.perf_counter() - start

    # Замер времени extract_trees
    n = len(tokens)
    start = time.perf_counter()
    # memo – кэш для рекурсии, создаём пустой словарь
    trees = extract_trees(
        0, n, root_symbol, tokens, dp,
        parser_engine.grammar, memo={}
    )
    time_extract = time.perf_counter() - start

    return time_cyk, time_extract


def run_benchmark(max_words=20, repeats=5, show_progress=True):
    """
    Запускает бенчмарк для длин слов от 1 до max_words.
    repeats – число повторений для каждой длины.
    Возвращает списки: lengths, cyk_times, extract_times (средние и стандартные отклонения).
    """
    # Инициализация парсера (один раз)
    grammar_path = Path(__file__).parent / "grammar.json"   # предполагаемый путь
    # Если у вас другой путь, измените или передайте через аргумент
    if not grammar_path.exists():
        raise FileNotFoundError(f"Грамматика не найдена: {grammar_path}. Укажите правильный путь.")

    grammar = load_grammar(str(grammar_path))
    grammar = binarize_grammar(fix_grammar(grammar))
    unary_idx, binary_idx = invert_grammar(grammar)

    # Создаём простой объект-контейнер для доступа к индексам и грамматике
    class Engine:
        pass
    engine = Engine()
    engine.unary_index = unary_idx
    engine.binary_index = binary_idx
    engine.grammar = grammar

    lengths = list(range(1, max_words + 1))
    cyk_means = []
    cyk_stds = []
    extract_means = []
    extract_stds = []

    # Прогресс-бар
    iterator = tqdm(lengths, desc="Обработка длин", disable=not show_progress)
    for L in iterator:
        sentence = generate_sentence(L)
        cyk_times = []
        extract_times = []
        for _ in range(repeats):
            t_cyk, t_ext = measure_parser(sentence, engine)
            cyk_times.append(t_cyk)
            extract_times.append(t_ext)
        cyk_means.append(statistics.mean(cyk_times))
        cyk_stds.append(statistics.stdev(cyk_times) if len(cyk_times) > 1 else 0.0)
        extract_means.append(statistics.mean(extract_times))
        extract_stds.append(statistics.stdev(extract_times) if len(extract_times) > 1 else 0.0)
        iterator.set_postfix(cyk=f"{cyk_means[-1]:.4f}s", ext=f"{extract_means[-1]:.4f}s")

    return lengths, cyk_means, cyk_stds, extract_means, extract_stds


def plot_results(lengths, cyk_means, cyk_stds, extract_means, extract_stds, save_path=None):
    """Строит графики зависимости времени от длины предложения."""
    plt.figure(figsize=(12, 5))

    # График для CYK
    plt.subplot(1, 2, 1)
    plt.errorbar(lengths, cyk_means, yerr=cyk_stds, fmt='ok-', capsize=3)
    plt.xlabel('Длина предложения (слов)')
    plt.ylabel('Время (сек)')
    plt.title('Построение таблицы CYK')
    plt.grid(True)

    # График для извлечения деревьев
    plt.subplot(1, 2, 2)
    plt.errorbar(lengths, extract_means, yerr=extract_stds, fmt='sk-', capsize=3)
    plt.ylabel('Время (сек)')
    plt.title('Построение деревьев разбора')
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Бенчмарк парсера (CYK + извлечение деревьев)')
    parser.add_argument('--max_words', type=int, default=20, help='Максимальная длина предложения (слов)')
    parser.add_argument('--repeats', type=int, default=5, help='Число прогонов для каждой длины')
    parser.add_argument('--no_progress', action='store_true', help='Отключить прогресс-бар')
    parser.add_argument('--save_plot', type=str, default=None, help='Путь для сохранения графика (PNG)')
    args = parser.parse_args()

    print("Запуск бенчмарка...")
    lengths, cyk_m, cyk_s, ext_m, ext_s = run_benchmark(
        max_words=args.max_words,
        repeats=args.repeats,
        show_progress=not args.no_progress
    )

    print("\nРезультаты (среднее ± std):")
    print("Длина\tCYK (с)\t\tДеревья (с)")
    for L, c_m, c_s, e_m, e_s in zip(lengths, cyk_m, cyk_s, ext_m, ext_s):
        print(f"{L:3d}\t{c_m:.6f} ± {c_s:.6f}\t{e_m:.6f} ± {e_s:.6f}")

    plot_results(lengths, cyk_m, cyk_s, ext_m, ext_s, save_path=args.save_plot)


if __name__ == "__main__":
    main()