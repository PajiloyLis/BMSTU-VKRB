import json

# Префикс вспомогательных нетерминалов при бинаризации (см. binarize_grammar).
BINARIZATION_AUX_PREFIX = "__AUX_"


def load_grammar(path):
    with open(path) as f:
        return json.load(f)

def binarize_grammar(gram):
    """
    Превращает правила длины >2 в несколько бинарных, вводя вспомогательные нетерминалы.
    Например, VP -> V NP PP  => VP -> V Z0, Z0 -> NP PP
    """
    new_gram = {lhs: [] for lhs in gram}
    aux_counter = 0
    for lhs, rules in gram.items():
        for rhs in rules:
            if len(rhs) <= 2:
                new_gram[lhs].append(rhs)
                continue
            # rhs = s0 s1 ... s_{k-1}, k >= 3: lhs -> s0 A0, A{i} -> s_{i+1} A{i+1}, A_{last} -> s_{k-2} s_{k-1}
            k = len(rhs)
            aux_names = [f"{BINARIZATION_AUX_PREFIX}{aux_counter + i}" for i in range(k - 2)]
            aux_counter += k - 2
            new_gram[lhs].append([rhs[0], aux_names[0]])
            for i in range(k - 3):
                new_gram.setdefault(aux_names[i], []).append([rhs[i + 1], aux_names[i + 1]])
            new_gram.setdefault(aux_names[k - 3], []).append([rhs[k - 2], rhs[k - 1]])
    return new_gram

def fix_grammar(gram):
    """
    Удаляем заведомо неверные зеркальные правила для PP и NP.
    Предлоги всегда слева от именной группы, и ИГ не может быть справа от предлога в линейном порядке.
    """
    # Правила для PP: только P NP, убираем NP P
    gram["PP"] = [r for r in gram["PP"] if r == ["P", "NP"]]
    # Для NP убираем варианты, где предлог/PP оказываются слева от N,
    # если только это не комплемент (N PP оставляем, NP CP оставляем)
    # Убираем явно ошибочные: ["PP", "N"], ["N", "Det"] - это нормально? 
    # Оставляем только те, что лингвистически оправданы:
    allowed_NP = [
        ["AP", "N"], ["N", "AP"],          # прилагательные могут быть с обеих сторон
        ["AP", "NP"],                     # прилагательное + ИГ (символическим числом 451)
        ["Det", "N"], ["N", "Det"],       # детерминанты иногда постпозитивны (люди эти)
        ["Q", "N"], ["N", "Q"],           # кванторы
        ["N", "N"],                       # имя + род. комплемент (симфониями возжигания)
        ["N", "NP"],                      # имя + ИГ-комплемент (симфониями возжигания и испепеления)
        ["N", "PP"],                      # комплемент справа
        ["N", "CP"],                      # придаточное справа
        ["N", "ConjAP"],                  # имя + запятая + определение / причастие (дирижера, управляющего …)
        ["Pron"], ["N"],                  # одиночные
        ["NP", "ConjNP"],                 # однородные ИГ через союз / запятую
        ["NP", "NP"],                     # асиндета: однородные ИГ подряд без запятой
        ["NP", "ConjCP"],                 # ИГ + запятая + придаточное (предвкушение того, что …; небо, которое …)
    ]
    gram["NP"] = [r for r in gram["NP"] if r in allowed_NP]
    return gram