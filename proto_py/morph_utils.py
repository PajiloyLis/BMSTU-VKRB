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

def _extract_gender(grammemes):
    if 'masc' in grammemes:
        return 'm'
    if 'femn' in grammemes:
        return 'f'
    if 'neut' in grammemes:
        return 'n'
    return None

def _extract_number(grammemes):
    if 'sing' in grammemes:
        return 'sg'
    if 'plur' in grammemes:
        return 'pl'
    return None

def _extract_case(grammemes):
    for case in ('nomn', 'gent', 'datv', 'accs', 'ablt', 'loct'):
        if case in grammemes:
            return case
    return None

def _extract_person(grammemes):
    if '1per' in grammemes:
        return '1'
    if '2per' in grammemes:
        return '2'
    if '3per' in grammemes:
        return '3'
    return None

def _extract_tense(grammemes):
    if 'past' in grammemes:
        return 'past'
    if 'pres' in grammemes:
        return 'pres'
    if 'futr' in grammemes:
        return 'fut'
    return None

def _extract_verb_features(grammemes, pos):
    """Возвращает словарь с признаками глагола/инфинитива."""
    feats = {}
    if pos in ("VERB", "INFN"):
        if "impf" in grammemes:
            feats["aspect"] = "impf"
        elif "perf" in grammemes:
            feats["aspect"] = "perf"
        if "tran" in grammemes:
            feats["trans"] = "tran"
        elif "intr" in grammemes:
            feats["trans"] = "intr"
        feats["verb_form"] = "infn" if pos == "INFN" else "fin"
    return feats


def _get_punctuation_features(word: str):
    """Возвращает признаки для знаков пунктуации или None."""
    if word == ",":
        return [{"pos": "Comma"}]
    if word == ";":
        return [{"pos": "Semicolon"}]
    if word == ":":
        return [{"pos": "Colon"}]
    return None

def _get_digit_features(word: str):
    """Возвращает признаки для цифр (квантор) или None."""
    if word.isdigit():
        return [{"pos": "Q"}]
    return None

def get_features(word):
    # Специальные токены
    punct_feats = _get_punctuation_features(word)
    if punct_feats is not None:
        return punct_feats

    # Цифры
    digit_feats = _get_digit_features(word)
    if digit_feats is not None:
        return digit_feats
    
    # Исключения
    lower_word = word.lower()
    if lower_word == "то":
        return [
            {"pos": "Pron", "gender": "n", "number": "sg", "case": "accs", "person": "3"},
            {"pos": "Pron", "gender": "n", "number": "sg", "case": "nomn", "person": "3"},
            {"pos": "C"},
        ]
    if lower_word == "всего":
        return [
            {"pos": "Adv"},
            {"pos": "A", "gender": "n", "number": "sg", "case": "gent"},
            {"pos": "C"},
        ]
    if lower_word == "её":
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
        g = p.tag.grammemes

        # Извлечение признаков
        gender = _extract_gender(g)
        if gender is not None:
            feats['gender'] = gender
        number = _extract_number(g)
        if number is not None:
            feats['number'] = number
        case = _extract_case(g)
        if case is not None:
            feats['case'] = case
        person = _extract_person(g)
        if person is not None:
            feats['person'] = person
        tense = _extract_tense(g)
        if tense is not None:
            feats['tense'] = tense

        # Глагольные признаки (вид, переходность, форма)
        verb_feats = _extract_verb_features(g, p.tag.POS)
        feats.update(verb_feats)

        results.append(feats)

    # Обработка неизвестных, но похожих на имена собственные
    if len(results) == 1 and results[0].get("pos") == "X":
        core = word.replace("ё", "е").replace("Ё", "Е")
        if len(core) >= 2 and core[0].isupper() and core.replace("-", "").isalpha():
            return [{"pos": "N", "number": "sg", "case": "nomn"}]
    return results

def preprocess_tokens(tokens: list[str]) -> list[tuple[str, list[dict[str, str]]]]:
    """
    Выполняет морфологический разбор для каждого токена.
    
    Args:
        tokens: список токенов (строк)
    
    Returns:
        Список такой же длины, каждый элемент — список словарей признаков
        (результат get_features для данного токена).
    """
    return [(token, get_features(token)) for token in tokens]
