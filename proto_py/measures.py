#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import statistics
import argparse
import csv
from datetime import datetime
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
    sentences = {
        1:  "Светает.",
        2:  "Кот спит.",
        3:  "Собака громко лает.",
        4:  "Дети играют во дворе.",
        5:  "Мама готовит вкусный горячий суп.",
        6:  "Старый пёс тихо лежит у двери.",
        7:  "Ветер дует сильно, листья летят быстро.",
        8:  "Пока брат читает книгу, сестра увлеченно рисует.",
        9:  "Солнце быстро зашло, стало темно сыро и тихо.",
        10: "Мы поздно пришли домой, быстро поели и легли спать.",
        11: "На улице идет дождь, дети сидят дома и смотрят мультики.",
        12: "Отец усердно работает в саду, мать варит варенье из спелых слив.",
        13: "Небо быстро потемнело, подул сильный резкий ветер, и начался холодный дождь.",
        14: "Мы сели в легковую машину, доехали до широкой реки и разбили маленький лагерь.",
        15: "Кот залез на высокое раскидистое дерево, сильно испугался большой высоты и громко протяжно мяукал.",
        16: "Бабушка пришла к нам домой, принесла свежий теплый пирог и осталась с нами на ужин.",  # уже было верным
17: "Дети с утра собрали грибы в лесу, мама их пожарила, и все дружно сели обедать.",
18: "Солнце уже давно зашло, на улице стало очень холодно, и мы все вместе решили вернуться домой.",
19: "Мы взяли рюкзаки, вышли из дома, дошли пешком до ближней станции и сели в самый первый поезд.",
20: "Летом мы ездили на юг к тёплому морю, каждый день купались там и загорали на горячем белом чистом песке.",
    }
    return sentences[length]
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
    print(sentence)
    return sentence


def measure_parser(text: str, parser_engine, root_symbol="IP") -> tuple[float, float, int]:
    """
    Выполняет полный разбор текста и возвращает (time_cyk, time_extract, tree_count).
    """
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
    trees = extract_trees(
        0, n, root_symbol, tokens, dp,
        parser_engine.grammar, memo={}
    )
    time_extract = time.perf_counter() - start

    return time_cyk, time_extract, len(trees)


def run_benchmark(max_words=20, repeats=5, show_progress=True):
    """
    Запускает бенчмарк для длин слов от 1 до max_words.
    repeats – число повторений для каждой длины.
    Возвращает список всех измерений (list of dict).
    """
    grammar_path = Path(__file__).parent / "grammar.json"
    if not grammar_path.exists():
        raise FileNotFoundError(f"Грамматика не найдена: {grammar_path}")

    grammar = load_grammar(str(grammar_path))
    grammar = binarize_grammar(fix_grammar(grammar))
    unary_idx, binary_idx = invert_grammar(grammar)

    class Engine:
        pass
    engine = Engine()
    engine.unary_index = unary_idx
    engine.binary_index = binary_idx
    engine.grammar = grammar

    lengths = list(range(1, max_words + 1))
    all_measurements = []

    iterator = tqdm(lengths, desc="Обработка длин", disable=not show_progress)
    for L in iterator:
        sentence = generate_sentence(L)
        for _ in range(repeats):
            t_cyk, t_ext, n_trees = measure_parser(sentence, engine)
            all_measurements.append({
                'length': L,
                'cyk_time': t_cyk,
                'extract_time': t_ext,
                'tree_count': n_trees
            })
        # опционально: выводить последние значения в прогресс-баре
        last = all_measurements[-1]
        iterator.set_postfix(cyk=f"{last['cyk_time']:.4f}s", ext=f"{last['extract_time']:.4f}s", trees=last['tree_count'])

    return all_measurements

def save_measurements_to_csv(measurements, csv_path):
    """
    Сохраняет список всех замеров в CSV файл (дозапись).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'length', 'cyk_time_sec', 'extract_trees_time_sec', 'tree_count'])

        for m in measurements:
            writer.writerow([
                timestamp,
                m['length'],
                f"{m['cyk_time']:.6f}",
                f"{m['extract_time']:.6f}",
                m['tree_count']
            ])


def plot_results(lengths, cyk_means, cyk_stds, extract_means, extract_stds, tree_counts, save_dir=None):
    """
    Строит отдельные графики зависимости времени от длины предложения.
    Сохраняет PDF в текущей рабочей директории с timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Нормировка extract на количество деревьев
    extract_means_per_tree = [
        m / k if k > 0 else 0.0
        for m, k in zip(extract_means, tree_counts)
    ]
    extract_stds_per_tree = [
        s / k if k > 0 else 0.0
        for s, k in zip(extract_stds, tree_counts)
    ]

    # --- График 1: Построение таблицы CYK ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(lengths, cyk_means, yerr=cyk_stds, fmt='ok-', capsize=3, linewidth=1.5, markersize=6)
    plt.xlabel('Количество словоформ в предложении', fontsize=12)
    plt.ylabel('Время (сек)', fontsize=12)
    plt.title('Построение таблицы составляющих', fontsize=14)
    plt.grid(True)
    
    cyk_filename = f"{timestamp}_cyk.pdf"
    cyk_path = Path.cwd() / cyk_filename
    plt.savefig(cyk_path, dpi=150, bbox_inches='tight')
    print(f"График CYK сохранён: {cyk_path}")
    plt.show()
    plt.close()

    # --- График 2: Среднее время построения одного дерева разбора ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(lengths, extract_means_per_tree, yerr=extract_stds_per_tree,
                 fmt='sk-', capsize=3, linewidth=1.5, markersize=6)
    plt.xlabel('Количество словоформ в предложении', fontsize=12)
    plt.ylabel('Среднее время на одно дерево (сек)', fontsize=12)
    plt.title('Построение деревьев разбора', fontsize=14)
    plt.grid(True)

    trees_filename = f"{timestamp}_trees.pdf"
    trees_path = Path.cwd() / trees_filename
    plt.savefig(trees_path, dpi=150, bbox_inches='tight')
    print(f"График деревьев сохранён: {trees_path}")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Бенчмарк парсера (CYK + извлечение деревьев)')
    parser.add_argument('--max_words', type=int, default=20, help='Максимальная длина предложения (слов)')
    parser.add_argument('--repeats', type=int, default=5, help='Число прогонов для каждой длины')
    parser.add_argument('--no_progress', action='store_true', help='Отключить прогресс-бар')
    parser.add_argument('--csv', type=str, default='benchmark_results.csv', help='Путь к CSV файлу для сохранения результатов')
    args = parser.parse_args()

    print("Запуск бенчмарка...")
    measurements = run_benchmark(
        max_words=args.max_words,
        repeats=args.repeats,
        show_progress=not args.no_progress
    )

    if not measurements:
        print("Нет данных.")
        return

    # Сохранение всех замеров в CSV
    csv_path = Path(args.csv)
    save_measurements_to_csv(measurements, csv_path)
    print(f"\nРезультаты (все замеры) сохранены в CSV: {csv_path}")

    # Агрегация для вывода в консоль и графиков
    grouped = {}
    for m in measurements:
        L = m['length']
        if L not in grouped:
            grouped[L] = {'cyk': [], 'ext': [], 'trees': []}
        grouped[L]['cyk'].append(m['cyk_time'])
        grouped[L]['ext'].append(m['extract_time'])
        grouped[L]['trees'].append(m['tree_count'])

    lengths = sorted(grouped.keys())
    cyk_means = [np.mean(grouped[L]['cyk']) for L in lengths]
    cyk_stds  = [np.std(grouped[L]['cyk'], ddof=1) if len(grouped[L]['cyk'])>1 else 0.0 for L in lengths]
    ext_means = [np.mean(grouped[L]['ext']) for L in lengths]
    ext_stds  = [np.std(grouped[L]['ext'], ddof=1) if len(grouped[L]['ext'])>1 else 0.0 for L in lengths]
    tree_counts = [int(np.mean(grouped[L]['trees'])) for L in lengths]

    # Нормированные значения для вывода в таблицу
    ext_m_per_tree = [m / k if k > 0 else 0.0 for m, k in zip(ext_means, tree_counts)]
    ext_s_per_tree = [s / k if k > 0 else 0.0 for s, k in zip(ext_stds, tree_counts)]

    print("\nРезультаты (среднее ± std по каждому length):")
    print(f"{'Длина':>5}  {'CYK (с)':>25}  {'Дерево (с/шт)':>25}  {'Деревьев':>10}")
    for L, c_m, c_s, e_m, e_s, n_t in zip(lengths, cyk_means, cyk_stds, ext_m_per_tree, ext_s_per_tree, tree_counts):
        print(f"{L:5d}  {c_m:.6f} ± {c_s:.6f}  {e_m:.6f} ± {e_s:.6f}  {n_t:10d}")

    # Построение графиков (используем функцию plot_results, но с полученными агрегатами)
    # Для совместимости с существующей plot_results передаём ей кортеж из (lengths, cyk_means, cyk_stds, ext_means, ext_stds, tree_counts)
    # Однако в старой plot_results используется нормировка внутри, поэтому просто передадим те же данные
    plot_results(lengths, cyk_means, cyk_stds, ext_means, ext_stds, tree_counts)

if __name__ == "__main__":
    main()