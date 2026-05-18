# main.py
from grammar import load_grammar, binarize_grammar, fix_grammar
from cyk import build_cyk_table
from morph_utils import tokenize_input
from tree_utils import extract_trees, tree_to_bracket, render_dot

def main():
    gram = load_grammar("grammar.json")
    gram = fix_grammar(gram)
    gram = binarize_grammar(gram)

    s = input()
    tokens = tokenize_input(s)
    print(tokens)
    n = len(tokens)
    dp = build_cyk_table(tokens, gram)

    # отладочная печать числа состояний в ячейках (dp[i][j] — отрезок [i, j))
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            cell = dp[i][j]
            if cell:
                print((i, j), {k: len(v) for k, v in cell.items()})
    if 'IP' not in dp[0][n]:
        print("Разбор не найден")
        return

    all_trees = extract_trees(0, n, 'IP', tokens, dp, gram, {})
    print(f"Найдено {len(all_trees)} деревьев")
    for idx, tree in enumerate(all_trees):
        print(f"===== Дерево {idx+1} =====")
        print(tree_to_bracket(tree))
        dot_filename = f"tree_{idx+1}"
        render_dot(tree, dot_filename, view=False)   # картинка сохраняется, но не открывается
        # если нужно открыть:
        # render_dot(tree, dot_filename, view=True)

if __name__ == "__main__":
    main()