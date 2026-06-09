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
    Загружает CSV и агрегирует данные по длинам предложений.
    Возвращает:
        lengths: список уникальных длин
        cyk_means, cyk_5, cyk_95: медиана и 5/95% времени CYK
        ext_per_tree_means, ext_per_tree_5, ext_per_tree_95: аналогично для времени на одно дерево
        ext_total_means, ext_total_5, ext_total_95: аналогично для суммарного времени
    """
    data_by_len = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = ['length', 'cyk_time_sec', 'extract_trees_time_sec', 'tree_count']
        if not all(col in reader.fieldnames for col in required):
            raise ValueError(f"CSV должен содержать колонки: {required}")
        for row in reader:
            length = int(row['length'])
            cyk = float(row['cyk_time_sec'])
            ext_total = float(row['extract_trees_time_sec'])
            trees = int(row['tree_count'])
            ext_per_tree = ext_total / trees if trees > 0 else 0.0
            if length not in data_by_len:
                data_by_len[length] = {'cyk': [], 'ext_per_tree': [], 'ext_total': []}
            data_by_len[length]['cyk'].append(cyk)
            data_by_len[length]['ext_per_tree'].append(ext_per_tree)
            data_by_len[length]['ext_total'].append(ext_total)

    lengths = sorted(data_by_len.keys())
    cyk_means, cyk_5, cyk_95 = [], [], []
    ext_per_tree_means, ext_per_tree_5, ext_per_tree_95 = [], [], []
    ext_total_means, ext_total_5, ext_total_95 = [], [], []

    for L in lengths:
        cyk_vals = np.array(data_by_len[L]['cyk'])
        ext_per_tree_vals = np.array(data_by_len[L]['ext_per_tree'])
        ext_total_vals = np.array(data_by_len[L]['ext_total'])

        cyk_means.append(np.median(cyk_vals))
        cyk_5.append(np.percentile(cyk_vals, 5) if len(cyk_vals) > 1 else cyk_means[-1])
        cyk_95.append(np.percentile(cyk_vals, 95) if len(cyk_vals) > 1 else cyk_means[-1])

        ext_per_tree_means.append(np.median(ext_per_tree_vals))
        ext_per_tree_5.append(np.percentile(ext_per_tree_vals, 5) if len(ext_per_tree_vals) > 1 else ext_per_tree_means[-1])
        ext_per_tree_95.append(np.percentile(ext_per_tree_vals, 95) if len(ext_per_tree_vals) > 1 else ext_per_tree_means[-1])

        ext_total_means.append(np.median(ext_total_vals))
        ext_total_5.append(np.percentile(ext_total_vals, 5) if len(ext_total_vals) > 1 else ext_total_means[-1])
        ext_total_95.append(np.percentile(ext_total_vals, 95) if len(ext_total_vals) > 1 else ext_total_means[-1])

    # Вывод статистики
    print("\nСтатистика по длинам предложений:")
    print(f"{'Len':>4} | {'CYK med (s)':>11} | {'5%':>6} | {'95%':>7} | {'Ext/tree med (s)':>15} | {'5%':>6} | {'95%':>7} | {'Ext total med (s)':>16} | {'5%':>6} | {'95%':>7}")
    print("-" * 110)
    for L, cm, c5, c95, epm, ep5, ep95, etm, et5, et95 in zip(lengths,
                                                               cyk_means, cyk_5, cyk_95,
                                                               ext_per_tree_means, ext_per_tree_5, ext_per_tree_95,
                                                               ext_total_means, ext_total_5, ext_total_95):
        print(f"{L:4d} | {cm:11.6f} | {c5:6.4f} | {c95:7.4f} | {epm:15.6f} | {ep5:6.4f} | {ep95:7.4f} | {etm:16.6f} | {et5:6.4f} | {et95:7.4f}")

    return (lengths,
            cyk_means, cyk_5, cyk_95,
            ext_per_tree_means, ext_per_tree_5, ext_per_tree_95,
            ext_total_means, ext_total_5, ext_total_95)


def best_poly_degree(x, y, max_degree=6):
    """Определяет степень полинома (от 1 до max_degree) по минимуму AIC."""
    n = len(x)
    best_deg, best_aic, best_coefs = 1, np.inf, None
    for deg in range(1, max_degree + 1):
        coefs = np.polyfit(x, y, deg)
        poly = np.poly1d(coefs)
        residuals = y - poly(x)
        mse = np.mean(residuals ** 2)
        aic = n * np.log(mse) + 2 * (deg + 1) if mse > 0 else -np.inf
        if aic < best_aic:
            best_aic, best_deg, best_coefs = aic, deg, coefs
    return best_deg, best_coefs


def fit_log_poly(x, y, max_degree=4):
    """Подгоняет полином к логарифму y. Возвращает степень, коэффициенты полинома (для log y)."""
    log_y = np.log(y)
    # Убираем возможные нули или отрицательные y (не должно быть, но на всякий случай)
    mask = np.isfinite(log_y)
    x_f = np.array(x)[mask]
    log_y_f = log_y[mask]
    if len(x_f) < 2:
        return 1, np.polyfit(x_f, log_y_f, 1)
    deg, coefs = best_poly_degree(x_f, log_y_f, max_degree=max_degree)
    return deg, coefs


def plot_results(lengths,
                 cyk_means, cyk_5, cyk_95,
                 ext_per_tree_means, ext_per_tree_5, ext_per_tree_95,
                 ext_total_means, ext_total_5, ext_total_95,
                 output_prefix=None):
    """Строит пять графиков: CYK, время/дерево (обычный и лог), суммарное время (обычный и лог)."""
    # Оптимальные степени для обычных (не логарифмических) графиков
    deg_cyk, coefs_cyk = best_poly_degree(lengths, cyk_means)
    deg_ep, coefs_ep = best_poly_degree(lengths, ext_per_tree_means)
    deg_et, coefs_et = best_poly_degree(lengths, ext_total_means)
    print("Коэффициенты CYK", *coefs_cyk);
    # Для логарифмических графиков подгоняем полином к log(время)
    deg_ep_log, coefs_ep_log = fit_log_poly(lengths, ext_per_tree_means, max_degree=4)
    deg_et_log, coefs_et_log = fit_log_poly(lengths, ext_total_means, max_degree=4)
    print("Коэффициенты log ext", *coefs_et_log)
    print(f"Оптимальная степень полинома для CYK: {deg_cyk}")
    print(f"Оптимальная степень полинома для времени на одно дерево: {deg_ep}")
    print(f"Оптимальная степень полинома для суммарного времени: {deg_et}")
    print(f"Оптимальная степень полинома для log(время/дерево): {deg_ep_log}")
    print(f"Оптимальная степень полинома для log(суммарное время): {deg_et_log}")

    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"{timestamp}_benchmark"

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    width = 0.6
    x_fit = np.linspace(min(lengths), max(lengths), 200)

    # ---- График 1: CYK ----
    plt.figure(figsize=(8, 6))
    yerr_low = [m - lo for m, lo in zip(cyk_means, cyk_5)]
    yerr_high = [hi - m for m, hi in zip(cyk_means, cyk_95)]
    plt.bar(lengths, cyk_means, width=width,
            yerr=[yerr_low, yerr_high], capsize=4,
            color='gray', edgecolor='black', label='Полученные данные')
    poly_cyk = np.poly1d(coefs_cyk)
    plt.plot(x_fit, poly_cyk(x_fit), 'k--', linewidth=2,
             label=f'Аппроксимация полиномом степени {deg_cyk}')
    plt.xticks(lengths)
    plt.xlabel('Количество словоформ в предложении', fontsize=12)
    plt.ylabel('Время, с', fontsize=12)
    plt.title('Медианное время построения таблицы составляющих', fontsize=14)
    plt.grid(True, linestyle='-')
    plt.legend()
    cyk_pdf = f"{output_prefix}_cyk_hist.pdf"
    plt.savefig(cyk_pdf, dpi=150, bbox_inches='tight')
    print(f"График CYK сохранён: {cyk_pdf}")
    plt.show()
    plt.close()

    # ---- График 2: время на одно дерево (обычный) ----
    plt.figure(figsize=(8, 6))
    yerr_low = [m - lo for m, lo in zip(ext_per_tree_means, ext_per_tree_5)]
    yerr_high = [hi - m for m, hi in zip(ext_per_tree_means, ext_per_tree_95)]
    plt.bar(lengths, ext_per_tree_means, width=width,
            yerr=[yerr_low, yerr_high], capsize=4,
            color='gray', edgecolor='black', label='Медианное время на дерево')
    poly_ep = np.poly1d(coefs_ep)
    plt.plot(x_fit, poly_ep(x_fit), 'k--', linewidth=2,
             label=f'Аппроксимация (степень {deg_ep})')
    plt.xticks(lengths)
    plt.xlabel('Количество словоформ в предложении', fontsize=12)
    plt.ylabel('Время восстановления одного дерева (сек)', fontsize=12)
    plt.title('Восстановление деревьев разбора (на одно дерево)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend()
    per_tree_pdf = f"{output_prefix}_trees_per_tree_hist.pdf"
    plt.savefig(per_tree_pdf, dpi=150, bbox_inches='tight')
    print(f"График времени на одно дерево сохранён: {per_tree_pdf}")
    plt.show()
    plt.close()

    # ---- График 3: суммарное время всех деревьев (обычный) ----
    plt.figure(figsize=(8, 6))
    yerr_low = [m - lo for m, lo in zip(ext_total_means, ext_total_5)]
    yerr_high = [hi - m for m, hi in zip(ext_total_means, ext_total_95)]
    plt.bar(lengths, ext_total_means, width=width,
            yerr=[yerr_low, yerr_high], capsize=4,
            color='gray', edgecolor='black', label='Полученные данные')
    poly_et = np.poly1d(coefs_et)
    # plt.plot(x_fit, poly_et(x_fit), 'k--', linewidth=2,
    #          label=f'Аппроксимация (степень {deg_et})')
    plt.xticks(lengths)
    plt.xlabel('Количество словоформ в предложении', fontsize=12)
    plt.ylabel('Время, с', fontsize=12)
    plt.title('Медианное время восстановления всех деревьев разбора', fontsize=14)
    plt.grid(True, linestyle='-')
    plt.legend()
    total_pdf = f"{output_prefix}_trees_total_hist.pdf"
    plt.savefig(total_pdf, dpi=150, bbox_inches='tight')
    print(f"График суммарного времени деревьев сохранён: {total_pdf}")
    plt.show()
    plt.close()

    # ---- График 4: время на одно дерево (логарифмическая шкала Y) ----
    plt.figure(figsize=(8, 6))
    # Преобразуем данные в логарифмы для отображения усов
    # Усы в логарифмическом масштабе: показываем [log(5%), log(95%)] или лучше исходные значения на логарифмической шкале.
    # Для barh с логарифмической шкалой передаём yerr в исходных единицах, а scale='log' сделает своё.
    # Однако планки погрешностей будут отображаться в логарифмическом масштабе корректно,
    # если мы передадим асимметричную ошибку в исходных единицах, а ось установим logarithmic.
    plt.bar(lengths, ext_per_tree_means, width=width,
            yerr=[yerr_low, yerr_high], capsize=4,
            color='gray', edgecolor='black', label='Медианное время на дерево')
    # Аппроксимация для логарифмических данных: строим exp(полином(log))
    poly_ep_log = np.poly1d(coefs_ep_log)
    y_fit_log = np.exp(poly_ep_log(x_fit))
    plt.plot(x_fit, y_fit_log, 'k--', linewidth=2,
             label=f'Аппроксимация (степень {deg_ep_log}) в логарифмической шкале')
    plt.xticks(lengths)
    plt.xlabel('Количество словоформ в предложении', fontsize=12)
    plt.ylabel('Время восстановления одного дерева (сек) — логарифмическая шкала', fontsize=12)
    plt.title('Восстановление деревьев разбора (на одно дерево, логарифмическая шкала)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.yscale('log')
    plt.legend()
    log_per_tree_pdf = f"{output_prefix}_trees_per_tree_log_hist.pdf"
    plt.savefig(log_per_tree_pdf, dpi=150, bbox_inches='tight')
    print(f"Логарифмический график времени на одно дерево сохранён: {log_per_tree_pdf}")
    plt.show()
    plt.close()

    # ---- График 5: суммарное время всех деревьев (логарифмическая шкала Y) ----
    plt.figure(figsize=(8, 6))
    # Используем те же yerr_low, yerr_high, что рассчитаны ранее для суммарного времени
    yerr_low_total = [m - lo for m, lo in zip(ext_total_means, ext_total_5)]
    yerr_high_total = [hi - m for m, hi in zip(ext_total_means, ext_total_95)]
    plt.bar(lengths, ext_total_means, width=width,
            yerr=[yerr_low_total, yerr_high_total], capsize=4,
            color='gray', edgecolor='black', label='Полученные данные')
    poly_et_log = np.poly1d(coefs_et_log)
    y_fit_log_total = np.exp(poly_et_log(x_fit))
    plt.plot(x_fit, y_fit_log_total, 'k--', linewidth=2,
             label=f'Аппроксимация полиномом степени {deg_et_log}')
    plt.xticks(lengths)
    plt.xlabel('Количество словоформ в предложении', fontsize=12)
    plt.ylabel('Время, с', fontsize=12)
    plt.title('Медианное время восстановления всех деревьев разбора', fontsize=14)
    plt.grid(True, linestyle='-')
    plt.yscale('log')
    plt.legend()
    log_total_pdf = f"{output_prefix}_trees_total_log_hist.pdf"
    plt.savefig(log_total_pdf, dpi=150, bbox_inches='tight')
    print(f"Логарифмический график суммарного времени деревьев сохранён: {log_total_pdf}")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Построение гистограмм производительности (обычных и логарифмических) из CSV')
    parser.add_argument('csv_file', type=str, help='Путь к CSV-файлу с результатами измерений')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Префикс для выходных PDF-файлов (без расширения)')
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Ошибка: файл {csv_path} не найден.")
        return

    print(f"Загрузка данных из {csv_path}...")
    (lengths,
     cyk_m, cyk_5, cyk_95,
     ep_m, ep_5, ep_95,
     et_m, et_5, et_95) = load_and_aggregate(str(csv_path))

    if not lengths:
        print("Нет данных для построения графиков.")
        return

    plot_results(lengths, cyk_m, cyk_5, cyk_95,
                 ep_m, ep_5, ep_95,
                 et_m, et_5, et_95,
                 output_prefix=args.output)


if __name__ == "__main__":
    main()