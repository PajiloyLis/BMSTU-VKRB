import pymorphy3
from json import loads, dumps

with open("translate_tags.json") as f:
    translator = loads(''.join(f.readlines()))
print(translator)

with open("grammar.json") as f:
    grammar = loads(''.join(f.readlines()))
print(grammar)

analyzer = pymorphy3.MorphAnalyzer()
s = input()
chain = s.split()
print(chain)
n=len(chain)
dp =[[set()]*(n+1) for i in range(n+1)]
# for i in range(n):
#     dp[i][i].add(translator[analyzer.parse(chain[i])[0].tag.POS])

for j in range(1, n+1):
    pos = translator[analyzer.parse(chain[j-1])[0].tag.POS]
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
                                dp[i][j]=key
                                break
for row in dp:
    print(row)

