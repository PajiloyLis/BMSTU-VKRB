#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_and_aggregate(csv_path: str):
    """
    Загружает CSV-файл с результатами измерений и агрегирует данные.
    Ожидаемые колонки: timestamp, length, cyk_time_sec, extract_trees_time_sec, tree_count
    Возвращает:
        lengths: список уникальных длин
        cyk_means, cyk_stds: среднее и std cyk_time_sec по каждой длине
        ext_means, ext_stds: среднее и std времени на одно дерево по каждой длине
        tree_counts (опционально)
    """
    data_by_len = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Проверяем наличие колонок
        required = ['length', 'cyk_time_sec', 'extract_trees_time_sec', 'tree_count']
        if not all(col in reader.fieldnames for col in required):
            raise ValueError(f"CSV должен содержать колонки: {required}")
        for row in reader:
            length = int(row['length'])
            cyk = float(row['cyk_time_sec'])
            ext = float(row['extract_trees_time_sec'])
            trees = int(row['tree_count'])
            # Время на одно дерево (если деревьев нет, ставим 0)
            ext_per_tree = ext / trees if trees > 0 else 0.0
            if length not in data_by_len:
                data_by_len[length] = {'cyk': [], 'ext_per_tree': []}
            data_by_len[length]['cyk'].append(cyk)
            data_by_len[length]['ext_per_tree'].append(ext_per_tree)

    # Сортируем по длине
    lengths = sorted(data_by_len.keys())
    cyk_means = []
    cyk_stds = []
    ext_means = []
    ext_stds = []
    for L in lengths:
        cyk_vals = data_by_len[L]['cyk']
        ext_vals = data_by_len[L]['ext_per_tree']
        cyk_sorted = np.sort(cyk_vals)
        ext_sorted = np.sort(ext_vals)
        cyk_means.append(np.median(cyk_vals))
        cyk_stds.append(np.percentile(cyk_vals, 0.95) if len(cyk_vals) > 1 else 0.0)
        ext_means.append(np.median(ext_vals))
        ext_stds.append(np.percentile(ext_vals, 0.95) if len(ext_vals) > 1 else 0.0)

    # Вывод статистики в консоль
    print("\nСтатистика по длинам предложений:")
    print(f"{'Length':>6} | {'CYK mean (s)':>12} | {'CYK std (s)':>12} | {'Ext/tree mean (s)':>16} | {'Ext/tree std (s)':>16}")
    print("-" * 75)
    for L, cm, cs, em, es in zip(lengths, cyk_means, cyk_stds, ext_means, ext_stds):
        print(f"{L:6d} | {cm:12.6f} | {cs:12.6f} | {em:16.6f} | {es:16.6f}")

    return lengths, cyk_means, cyk_stds, ext_means, ext_stds

def best_poly_degree(x, y, max_degree=10):
    """
    Определяет степень полинома (от 1 до max_degree),
    которая минимизирует AIC.
    
    Параметры:
        x, y: списки или массивы данных
        max_degree: максимальная степень для проверки
    
    Возвращает:
        best_deg: выбранная степень
        coefs: коэффициенты полинома этой степени
    """
    n = len(x)
    best_deg = 1
    best_aic = np.inf
    best_coefs = None
    
    for deg in range(1, max_degree + 1):
        # Подгоняем полином
        coefs = np.polyfit(x, y, deg)
        poly = np.poly1d(coefs)
        y_pred = poly(x)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        if mse == 0:
            aic = -np.inf
        else:
            # AIC для регрессии: n * ln(MSE) + 2 * k, где k = deg+1
            aic = n * np.log(mse) + 2 * (deg + 1)
        
        if aic < best_aic:
            best_aic = aic
            best_deg = deg
            best_coefs = coefs
    
    return best_deg, best_coefs

def plot_results(lengths, cyk_means, cyk_stds, ext_means, ext_stds, output_prefix=None):
    """
    Строит два графика (CYK и дерево) и сохраняет в PDF.
    Если output_prefix не указан, формирует имя на основе временной метки.
    """
    
    deg_cyk, coefs_cyk = best_poly_degree(lengths, cyk_means, max_degree=4)
    deg_ext, coefs_ext = best_poly_degree(lengths, ext_means, max_degree=4)

    print(f"Оптимальная степень для CYK: {deg_cyk}")
    print(f"Оптимальная степень для извлечения деревьев: {deg_ext}")
    
    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"{timestamp}_benchmark"

    # ----- График 1: Время построения таблицы CYK -----
    plt.figure(figsize=(8, 6))
    plt.errorbar(lengths, cyk_means, yerr=cyk_stds, fmt='ok-', capsize=3, linewidth=1.5, markersize=6, label='Экспериментальные данные')
    
    # Аппроксимация полиномом 3-й степени для CYK
    coefs_cyk = np.polyfit(lengths, cyk_means, 3)
    poly_cyk = np.poly1d(coefs_cyk)
    n_fit = np.linspace(min(lengths), max(lengths), 100)
    plt.plot(n_fit, poly_cyk(n_fit), 'k--', linewidth=1.5, label=f'Аппроксимация O(n³)\n{coefs_cyk[0]:.2e}n³ + {coefs_cyk[1]:.2e}n² + {coefs_cyk[2]:.2e}n + {coefs_cyk[3]:.2e}')
    
    plt.xticks(lengths)
    plt.xlabel('Количество словоформ в предложении', fontsize=12)
    plt.ylabel('Время (сек)', fontsize=12)
    plt.title('Построение таблицы составляющих', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    cyk_pdf = f"{output_prefix}_cyk.pdf"
    plt.savefig(cyk_pdf, dpi=150, bbox_inches='tight')
    print(f"График CYK сохранён: {cyk_pdf}")
    plt.legend()
    plt.show()
    plt.close()

    # ----- График 2: Среднее время построения одного дерева разбора -----
    plt.figure(figsize=(8, 6))
    plt.errorbar(lengths, ext_means, yerr=ext_stds, fmt='sk-', capsize=3, linewidth=1.5, markersize=6, label='Экспериментальные данные')
    
    # Аппроксимация полиномом 2-й степени (можно и 3-й, смотрите по данным)
    coefs_ext = np.polyfit(lengths, ext_means, 4)  # или 3, если нужно
    poly_ext = np.poly1d(coefs_ext)
    plt.plot(n_fit, poly_ext(n_fit), 'k--', linewidth=1.5, label=f'Аппроксимация O(n⁴)\n{coefs_ext[0]:.2e}n⁴ + {coefs_ext[1]:.2e}n³ + {coefs_ext[2]:.2e}n² + {coefs_ext[3]:.2e}n + {coefs_ext[4]:.2e}')
    
    plt.xticks(lengths)
    plt.xlabel('Количество словоформ в предложении', fontsize=12)
    plt.ylabel('Среднее время на одно дерево (сек)', fontsize=12)
    plt.title('Построение деревьев разбора', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    trees_pdf = f"{output_prefix}_trees.pdf"
    plt.savefig(trees_pdf, dpi=150, bbox_inches='tight')
    print(f"График деревьев сохранён: {trees_pdf}")
    plt.legend()
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Построение графиков производительности из CSV-файла')
    parser.add_argument('csv_file', type=str, help='Путь к CSV-файлу с результатами измерений')
    parser.add_argument('--output', '-o', type=str, default=None, help='Префикс для выходных PDF-файлов (без расширения)')
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Ошибка: файл {csv_path} не найден.")
        return

    print(f"Загрузка данных из {csv_path}...")
    lengths, cyk_m, cyk_s, ext_m, ext_s = load_and_aggregate(str(csv_path))

    if not lengths:
        print("Нет данных для построения графиков.")
        return

    plot_results(lengths, cyk_m, cyk_s, ext_m, ext_s, output_prefix=args.output)


if __name__ == "__main__":
    main()