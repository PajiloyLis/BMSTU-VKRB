# tree_utils.py
import subprocess
import os
import platform

from morph_utils import TERMINAL_TAGS
from cyk import (
    agreement_check,
    agreement_check_unary,
    dict_from_frozenset,
    merge_features,
    merge_features_unary,
)
from grammar import BINARIZATION_AUX_PREFIX


def is_binarization_aux(tag):
    """Вспомогательный нетерминал цепочки бинаризации (не показываем в скобках/DOT)."""
    return isinstance(tag, str) and tag.startswith(BINARIZATION_AUX_PREFIX)

def format_feats(feats):
    """Отрисовка признаков в короткую строку (узлы дерева: DOT, скобочная запись)."""
    if not feats:
        return ""
    parts = []
    order = (
        "gender",
        "number",
        "case",
        "person",
        "tense",
        "aspect",
        "trans",
        "verb_form",
    )
    for key in order:
        if key in feats and feats[key] is not None:
            parts.append(str(feats[key]))
    return ",".join(parts)

def extract_trees(i, j, symbol, word_chain, dp, grammar, memo, target_feat=None):
    """
    Возвращает список деревьев для указанного symbol на отрезке [i,j).
    Каждое дерево имеет структуру:
        {'tag': str, 'feats': dict, 'children': list | 'word': str}
    Если target_feat задано (frozenset), возвращаются только деревья с таким набором признаков.
    """
    # Получаем все наборы признаков для данного symbol в ячейке
    feats_in_cell = dp[i][j].get(symbol, set())
    if target_feat is not None and target_feat not in feats_in_cell:
        return []   # нет подходящих признаков

    # Мемоизация: ключ – (i, j, symbol, target_feat)
    memo_key = (i, j, symbol, target_feat)
    if memo_key in memo:
        return memo[memo_key]

    trees = []

    # 1. Терминальный случай (длина 1)
    if j - i == 1:
        # Если symbol является терминальным тегом (N, V, ...), создаём лист
        if symbol in TERMINAL_TAGS:
            word = word_chain[i]
            for fs in feats_in_cell:
                if target_feat is not None and fs != target_feat:
                    continue
                feats_dict = dict_from_frozenset(fs)
                trees.append({
                    'tag': symbol,
                    'feats': feats_dict,
                    'word': word
                })
        # 2. Унарные правила: symbol -> X (где X – нетерминал или терминал)
        # Обрабатываем как "symbol выведен из X через правило длины 1".
        # В DP мы копировали признаки, поэтому для каждого fs из feats_in_cell
        # существует такое же fs у правой части правила.
        for lhs, rules in grammar.items():
            if lhs != symbol:
                continue
            for rhs in rules:
                if len(rhs) != 1:
                    continue
                child_sym = rhs[0]
                # Для каждого набора признаков fs текущего symbol
                for fs in feats_in_cell:
                    if target_feat is not None and fs != target_feat:
                        continue
                    # Извлекаем все поддеревья для child_sym, которые имеют ТОТ ЖЕ набор признаков fs
                    child_trees = extract_trees(i, j, child_sym, word_chain, dp, grammar, memo, target_feat=fs)
                    for child in child_trees:
                        feats_dict = dict_from_frozenset(fs)
                        trees.append({
                            'tag': symbol,
                            'feats': feats_dict,
                            'children': [child]
                        })
        # терминалы уже обработаны выше, унарные правила добавят обертки.
        memo[memo_key] = trees
        return trees

    # 3. Бинарные правила
    for k in range(i+1, j):
        left_cell = dp[i][k]
        right_cell = dp[k][j]
        for lhs, rules in grammar.items():
            if lhs != symbol:
                continue
            for rhs in rules:
                if len(rhs) != 2:
                    continue
                A, B = rhs[0], rhs[1]
                if A not in left_cell or B not in right_cell:
                    continue
                for featA in left_cell[A]:
                    for featB in right_cell[B]:
                        if not agreement_check(
                            lhs,
                            A,
                            featA,
                            B,
                            featB,
                            tokens=word_chain,
                            span_left=(i, k),
                            span_right=(k, j),
                        ):
                            continue
                        merged = merge_features(lhs, A, featA, B, featB)
                        if merged not in feats_in_cell:
                            continue
                        if target_feat is not None and merged != target_feat:
                            continue
                        # рекурсивно строим левое и правое поддеревья
                        left_trees = extract_trees(i, k, A, word_chain, dp, grammar, memo, target_feat=featA)
                        right_trees = extract_trees(k, j, B, word_chain, dp, grammar, memo, target_feat=featB)
                        for lt in left_trees:
                            for rt in right_trees:
                                feats_dict = dict_from_frozenset(merged)
                                trees.append({
                                    'tag': lhs,
                                    'feats': feats_dict,
                                    'children': [lt, rt]
                                })

    # 3b. Унарные правила на отрезке длины > 1 (например IP -> VP без подлежащего)
    if j - i >= 2:
        for lhs, rules in grammar.items():
            if lhs != symbol:
                continue
            for rhs in rules:
                if len(rhs) != 1:
                    continue
                child_sym = rhs[0]
                if child_sym not in dp[i][j]:
                    continue
                for cfs in dp[i][j][child_sym]:
                    if not agreement_check_unary(lhs, child_sym, cfs):
                        continue
                    merged = merge_features_unary(lhs, child_sym, cfs)
                    if merged not in feats_in_cell:
                        continue
                    if target_feat is not None and merged != target_feat:
                        continue
                    child_trees = extract_trees(
                        i, j, child_sym, word_chain, dp, grammar, memo, target_feat=cfs
                    )
                    for child in child_trees:
                        feats_dict = dict_from_frozenset(merged)
                        trees.append(
                            {
                                "tag": lhs,
                                "feats": feats_dict,
                                "children": [child],
                            }
                        )

    memo[memo_key] = trees
    return trees


def _bracket_pieces(node):
    """Фрагменты слева направо; узлы бинаризации разворачиваются без обёртки."""
    if 'word' in node:
        return [node['word']]
    if is_binarization_aux(node['tag']):
        parts = []
        for ch in node.get('children', []):
            parts.extend(_bracket_pieces(ch))
        return parts
    tag = node['tag']
    feats = format_feats(node.get('feats', {}))
    label = f"{tag}[{feats}]" if feats else tag
    inner = []
    for ch in node.get('children', []):
        inner.extend(_bracket_pieces(ch))
    return [f"{label}({' '.join(inner)})"]


def tree_to_bracket(node):
    """Рекурсивное преобразование в скобочную строку."""
    parts = _bracket_pieces(node)
    if len(parts) == 1:
        return parts[0]
    return ' '.join(parts)


def _dot_escape(text: str) -> str:
    """Экранирование для label=\"...\" в DOT."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def tree_to_dot(node, parent_id=None, nodes=None, edges=None):
    """Генерация DOT-описания дерева. Возвращает кортеж (nodes, edges)."""
    if nodes is None:
        nodes = []
        edges = []
    if is_binarization_aux(node.get('tag')) and node.get('children'):
        for child in node['children']:
            tree_to_dot(child, parent_id, nodes, edges)
        return nodes, edges
    node_id = id(node)
    tag = node['tag']
    feats = format_feats(node.get('feats', {}))
    label = f"{tag}\n{feats}" if feats else tag
    if 'word' in node:
        label = f"{node['word']}\n({tag})\n{feats}" if feats else f"{node['word']}\n({tag})"
    nodes.append(f'  {node_id} [label="{_dot_escape(label)}"];')
    if parent_id is not None:
        edges.append(f'  {parent_id} -> {node_id};')
    for child in node.get('children', []):
        tree_to_dot(child, node_id, nodes, edges)
    return nodes, edges


def render_dot(tree, filename="tree", view=True):
    """Сохраняет дерево в DOT-файл и при необходимости открывает картинку."""
    nodes, edges = tree_to_dot(tree)
    dot = "digraph Tree {\n"
    dot += "\n".join(nodes) + "\n"
    dot += "\n".join(edges) + "\n}"
    dot_path = f"{filename}.dot"
    with open(dot_path, "w") as f:
        f.write(dot)
    # конвертация через graphviz (требуется установленный dot)
    png_path = f"{filename}.png"
    try:
        subprocess.run(
            ["dot", "-Tpng", dot_path, "-o", png_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            f"Предупреждение: утилита Graphviz «dot» не найдена в PATH; "
            f"сохранён только {dot_path}"
        )
        return
    except subprocess.CalledProcessError as e:
        print(
            f"Предупреждение: «dot» завершился с ошибкой; {dot_path} сохранён. "
            f"{(e.stderr or '').strip()}"
        )
        return
    if view:
        if platform.system() == "Darwin":
            subprocess.run(["open", png_path])
        elif platform.system() == "Windows":
            os.startfile(png_path)
        else:
            subprocess.run(["xdg-open", png_path])