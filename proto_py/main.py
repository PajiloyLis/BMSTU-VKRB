import pymorphy3
from json import loads, dumps
from collections import defaultdict

with open("translate_tags.json") as f:
    translator = loads(''.join(f.readlines()))

with open("grammar.json") as f:
    grammar = loads(''.join(f.readlines()))
    
analyzer = pymorphy3.MorphAnalyzer()
s = input()
chain = s.split()
n=len(chain)
dp = [[set() for _ in range(n+1)] for _ in range(n+1)]

for j in range(1, n+1):
    for tag in analyzer.parse(chain[j-1]):
        pos = translator[tag.tag.POS]
        dp[j-1][j].add(pos)
        for key, value_list in grammar.items():
            for value in value_list:
                if pos in value and len(value)==1:
                    dp[j-1][j].add(key)
    for i in range(j-2, -1, -1):
        for k in range(i+1, j):
            for key, value_list in grammar.items():
                for value in value_list:
                    for tag_left in dp[i][k]:
                        for tag_right in dp[k][j]:
                            if tag_left in value and tag_right in value and len(value) == 2:
                                dp[i][j].add(key)
for row in dp:
    print(row)

grammar_index = defaultdict(list)
for key, rhs_list in grammar.items():
    for rhs in rhs_list:
        grammar_index[key].append(rhs)

def extract_trees(i, j, symbol, words, dp, grammar, terminal_tags, memo):
    """
    Возвращает список всех деревьев для заданного symbol на отрезке [i, j).
    """
    # Используем мемоизацию, чтобы не пересчитывать одинаковые вызовы
    key = (i, j, symbol)
    if key in memo:
        return memo[key]

    trees = []

    # 1. Терминальный случай (отрезок из одного слова)
    if j - i == 1:
        if symbol in terminal_tags:
            trees.append((symbol, words[i]))  # лист дерева: (тег, слово)
        else:
            # symbol выведен через правило длины 1 из какого-то терминала
            for rhs in grammar_index.get(symbol, []):
                if len(rhs) == 1 and rhs[0] in dp[i][j]:
                    # рекурсивно развернуть
                    for sub in extract_trees(i, j, rhs[0], words, dp, grammar, terminal_tags, memo):
                        trees.append((symbol, sub))
        memo[key] = trees
        return trees

    # 2. Бинарные правила (отрезок из 2+ слов)
    for k in range(i+1, j):
        for rhs in grammar_index.get(symbol, []):
            if len(rhs) != 2:
                continue
            left_sym, right_sym = rhs[0], rhs[1]
            if left_sym in dp[i][k] and right_sym in dp[k][j]:
                left_trees = extract_trees(i, k, left_sym, words, dp, grammar, terminal_tags, memo)
                right_trees = extract_trees(k, j, right_sym, words, dp, grammar, terminal_tags, memo)
                for lt in left_trees:
                    for rt in right_trees:
                        trees.append((symbol, [lt, rt]))
    memo[key] = trees
    return trees

# Множество терминальных тегов (то, что выдаёт pymorphy после трансляции)
terminal_tags = set(translator.values()) - {None}  # None если был проигнорирован

# Мемоизация
memo = {}


words = chain          # список токенов
all_trees = []
if 'IP' in dp[0][n]:
    all_trees = extract_trees(0, n, 'IP', words, dp, grammar, terminal_tags, memo)

print(f"Найдено {len(all_trees)} деревьев")
# Для просмотра можно напечатать вложенную структуру
for i, tree in enumerate(all_trees):
    print(f"Дерево {i+1}: {tree}")