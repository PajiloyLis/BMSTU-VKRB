import re

import pymorphy3

morph = pymorphy3.MorphAnalyzer()

# Запятая между цифрами не выделяется отдельным токеном (остаётся внутри «3,14»).
_DECIMAL_COMMA_PROTECT = re.compile(r"(\d),(\d)")

# Снятие пунктуации с краёв токена (в т.ч. типографские кавычки и тире)
_PUNCT_EDGES = re.compile(
    r"^[\s\u00A0\.!?;:\"'()\[\]{}«»„“”…\-—–]+|[\s\u00A0\.!?;:\"'()\[\]{}«»„“”…\-—–]+$",
    re.UNICODE,
)


def strip_punctuation_edges(word: str) -> str:
    if not word:
        return ""
    return _PUNCT_EDGES.sub("", word.strip())


def tokenize_input(line: str) -> list[str]:
    """
    Гибрид: запятая — отдельный токен «,» для Conj* в грамматике; остальная пунктуация
    с краёв слова снимается. Запятая между цифрами (3,14) не режется на токены.
    Точка с запятой — отдельный токен «;» для склейки клауз (ConjIP).
    Двоеточие — отдельный токен «:» (пояснение / вторая часть после ИГ).
    """
    s = _DECIMAL_COMMA_PROTECT.sub(lambda m: f"{m.group(1)}\u241c{m.group(2)}", line)
    s = s.replace(";", " ; ").replace(":", " : ").replace(",", " , ")
    out = []
    for w in s.split():
        if w == ",":
            out.append(",")
        elif w == ";":
            out.append(";")
        elif w == ":":
            out.append(":")
        else:
            t = strip_punctuation_edges(w).replace("\u241c", ",")
            if t:
                out.append(t)
    return out

# Отсечка по вероятности разбора (pymorphy3: поле Parse.score, ~0..1)
MIN_PARSE_SCORE = 0.1

# Преобразование POS в наши обозначения
POS_MAP = {
    "NOUN": "N", "ADJS": "A", "ADJF": "A", "COMP": "Adv",
    "VERB": "V", "INFN": "V", "PRTF": "A", "PRTS": "V",
    "GRND": "Adv", "ADVB": "Adv", "PRED": "Adv",
    "PREP": "P", "CONJ": "C", "PRCL": "C", "NPRO": "Pron", "NUMR": "Q"
}
# Терминалы pymorphy + запятая как отдельный токен (см. tokenize_input).
TERMINAL_TAGS = set(POS_MAP.values()) - {None} | {"Comma", "Semicolon", "Colon"}


def get_features(word):
    """
    Возвращает список словарей с признаками для каждого разбора слова.
    Каждый словарь имеет ключи 'pos' (наш тег) и набор граммем:
    род, число, падеж, лицо, время; для глаголов и инфинитивов — aspect, trans, verb_form.
    Учитываются только разборы с Parse.score >= MIN_PARSE_SCORE;
    если все ниже порога, берётся один лучший (первый в порядке pymorphy).
    """
    if word == ",":
        return [{"pos": "Comma"}]
    if word == ";":
        return [{"pos": "Semicolon"}]
    if word == ":":
        return [{"pos": "Colon"}]
    # Чистые цифры — как квантор (число 451), иначе pymorphy даёт X и ломает N Q.
    if word.isdigit():
        return [{"pos": "Q"}]
    if word.lower() == "то":
        return [
            {"pos": "Pron", "gender": "n", "number": "sg", "case": "accs", "person": "3"},
            {"pos": "Pron", "gender": "n", "number": "sg", "case": "nomn", "person": "3"},
            {"pos": "C"},
        ]
    if word.lower() == "всего":
        return [
            {"pos": "Adv"},
            {"pos": "A", "gender": "n", "number": "sg", "case": "gent"},
            {"pos": "C"},
        ]
    if word.lower() == "её":
        return [
            {"pos": "Pron", "gender": "f", "number": "sg", "case": "accs", "person": "3"},
            {"pos": "Pron", "gender": "f", "number": "sg", "case": "gent", "person": "3"},
            {"pos": "A", "gender": "f", "number": "sg", "case": "accs"},
            {"pos": "A", "gender": "f", "number": "sg", "case": "gent"},
        ]

    parses = morph.parse(word)
    filtered = [p for p in parses if p.score >= MIN_PARSE_SCORE]
    if not filtered:
        filtered = parses[:1]

    results = []
    for p in filtered:
        feats = {'pos': POS_MAP.get(p.tag.POS, 'X')}
        # Извлекаем нужные граммемы
        g = p.tag.grammemes
        # Род, число, падеж
        if 'masc' in g: feats['gender'] = 'm'
        elif 'femn' in g: feats['gender'] = 'f'
        elif 'neut' in g: feats['gender'] = 'n'
        # число
        if 'sing' in g: feats['number'] = 'sg'
        elif 'plur' in g: feats['number'] = 'pl'
        # падеж
        for case in ['nomn','gent','datv','accs','ablt','loct']:
            if case in g:
                feats['case'] = case
                break
        # лицо
        if '1per' in g: feats['person'] = '1'
        elif '2per' in g: feats['person'] = '2'
        elif '3per' in g: feats['person'] = '3'
        # время
        if 'past' in g: feats['tense'] = 'past'
        elif 'pres' in g: feats['tense'] = 'pres'
        elif 'futr' in g: feats['tense'] = 'fut'
        # глагол / инфинитив (OpenCorpora: tran/intr, impf/perf)
        if p.tag.POS in ("VERB", "INFN"):
            if "impf" in g:
                feats["aspect"] = "impf"
            elif "perf" in g:
                feats["aspect"] = "perf"
            if "tran" in g:
                feats["trans"] = "tran"
            elif "intr" in g:
                feats["trans"] = "intr"
            feats["verb_form"] = "infn" if p.tag.POS == "INFN" else "fin"
        results.append(feats)
    # pymorphy: UNKN → pos X; для имён/топонимов вроде «Ури» иначе не собрать NP NP + сказуемое.
    if len(results) == 1 and results[0].get("pos") == "X":
        core = word.replace("ё", "е").replace("Ё", "Е")
        if len(core) >= 2 and core[0].isupper() and core.replace("-", "").isalpha():
            return [{"pos": "N", "number": "sg", "case": "nomn"}]
    return results