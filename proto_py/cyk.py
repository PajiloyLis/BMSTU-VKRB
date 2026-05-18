import json
from pathlib import Path

from grammar import BINARIZATION_AUX_PREFIX
from morph_utils import get_features


def _is_binarization_aux(lhs):
    return isinstance(lhs, str) and lhs.startswith(BINARIZATION_AUX_PREFIX)

_PREP_PATH = Path(__file__).resolve().parent / "prepositions.json"
with _PREP_PATH.open(encoding="utf-8") as _f:
    _raw_prep = json.load(_f)
# ключи — фразы предлогов в нижнем регистре (как в разметке)
PREPOSITION_GOVERNMENT = {k.lower().strip(): tuple(v) for k, v in _raw_prep.items()}


def _unary_closure_round(dp, grammar, tokens, n):
    """
    Один полный проход унарных правил lhs -> sym по всем отрезам.
    Возвращает True, если добавились новые состояния.
    """
    changed = False
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            for lhs, rules in grammar.items():
                for rhs in rules:
                    if len(rhs) != 1:
                        continue
                    sym = rhs[0]
                    if sym not in dp[i][j]:
                        continue
                    for feat in list(dp[i][j][sym]):
                        if not agreement_check_unary(lhs, sym, feat):
                            continue
                        new_feat = merge_features_unary(lhs, sym, feat)
                        bucket = dp[i][j].setdefault(lhs, set())
                        before = len(bucket)
                        bucket.add(new_feat)
                        if len(bucket) > before:
                            changed = True
    return changed


def build_cyk_table(tokens, grammar):
    n = len(tokens)
    # dp[i][j] -> dict {nonterm: set of frozenset features}
    dp = [[dict() for _ in range(n+1)] for _ in range(n+1)]
    
    # Инициализация (j-1, j)
    for j in range(1, n+1):
        word = tokens[j-1]
        features_list = get_features(word)   # список словарей
        for feat in features_list:
            pos = feat['pos']
            # Добавляем терминал
            feat_set = frozenset(feat.items())
            dp[j-1][j].setdefault(pos, set()).add(feat_set)
            # Унарные правила
            for lhs, rules in grammar.items():
                for rhs in rules:
                    if len(rhs) == 1 and rhs[0] == pos:
                        dp[j-1][j].setdefault(lhs, set()).add(feat_set)
    # Унарные из терминалов (NP <- N и т.д.) — до бинарных длин >1
    while _unary_closure_round(dp, grammar, tokens, n):
        pass

    # Заполнение для более длинных отрезков: после каждой длины — унарное замыкание,
    # иначе CP <- C IP не видит IP <- VP на уже построенном VP (напр. «что сейчас произойдет»).
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            for k in range(i + 1, j):
                left_cell = dp[i][k]
                right_cell = dp[k][j]
                for lhs, rules in grammar.items():
                    for rhs in rules:
                        if len(rhs) != 2:
                            continue
                        symA, symB = rhs[0], rhs[1]
                        if symA in left_cell and symB in right_cell:
                            for featA in left_cell[symA]:
                                for featB in right_cell[symB]:
                                    if agreement_check(
                                        lhs,
                                        symA,
                                        featA,
                                        symB,
                                        featB,
                                        tokens=tokens,
                                        span_left=(i, k),
                                        span_right=(k, j),
                                    ):
                                        new_feat = merge_features(lhs, symA, featA, symB, featB)
                                        dp[i][j].setdefault(lhs, set()).add(new_feat)
        while _unary_closure_round(dp, grammar, tokens, n):
            pass
    return dp

def dict_from_frozenset(fs):
    return dict(fs)


def agreement_check_unary(lhs, sym, feat_fs):
    """
    Согласование для унарных правил lhs -> sym (после бинарного заполнения DP).
    IP -> VP: без подлежащего; инфинитив одним VP не считаем полным предложением.
    """
    d = dict_from_frozenset(feat_fs)
    if lhs == "IP" and sym == "VP":
        if d.get("verb_form") == "infn":
            return False
        return True
    if lhs == "IP" and sym in ("AdvP", "AP"):
        return True
    return True


def merge_features_unary(lhs, sym, feat_fs):
    """Признаки узла lhs при унарном правиле lhs -> sym."""
    if lhs == "IP" and sym in ("VP", "AdvP", "AP"):
        return feat_fs
    return feat_fs


def _agree_if_both(key, a, b):
    """Совпадение по полю, только если оба значения заданы (не None)."""
    va, vb = a.get(key), b.get(key)
    if va is not None and vb is not None and va != vb:
        return False
    return True


def _np_modifier_head_agree(mod_feats, head_feats):
    """Имя + модификатор (AP, Det): род/число/падеж согласованы, если оба значения есть."""
    for key in ("gender", "number", "case"):
        if not _agree_if_both(key, mod_feats, head_feats):
            return False
    return True


def _pp_prep_np_agree(prep_phrase, np_feats):
    """Падеж ИГ справа должен входить в список управления предлога (prepositions.json)."""
    key = prep_phrase.lower().strip()
    if key not in PREPOSITION_GOVERNMENT:
        return True
    allowed = PREPOSITION_GOVERNMENT[key]
    c = np_feats.get("case")
    if c is None:
        return True
    return c in allowed


def _np_coord_agree(a, b):
    """Однородные ИГ: падеж совпадает, если задан у обоих."""
    return _agree_if_both("case", a, b)


def _vp_coord_agree(a, b):
    """Однородные сказуемые: время, число, лицо — если заданы у обоих."""
    for key in ("tense", "number", "person"):
        if not _agree_if_both(key, a, b):
            return False
    return True


def _coord_np_merge_dict(a, b):
    """Признаки сочинённого NP (число мн. при двух ед., общий падеж)."""
    out = {}
    an, bn = a.get("number"), b.get("number")
    if an == "sg" and bn == "sg":
        out["number"] = "pl"
    elif an and bn and an != bn:
        out["number"] = "pl"
    else:
        out["number"] = an or bn
    ag, bg = a.get("gender"), b.get("gender")
    if ag and bg and ag == bg:
        out["gender"] = ag
    elif ag and not bg:
        out["gender"] = ag
    elif bg and not ag:
        out["gender"] = bg
    for key in ("case", "person"):
        va, vb = a.get(key), b.get(key)
        if va is not None:
            out[key] = va
        elif vb is not None:
            out[key] = vb
    pos = "Pron" if (a.get("pos") == "Pron" or b.get("pos") == "Pron") else "N"
    out["pos"] = pos
    return out


def _coord_ap_merge_dict(a, b):
    """Признаки сочинённого определения (AP): общий род/число/падеж для согласования с N."""
    out = {}
    for key in ("gender", "number", "case", "person", "tense"):
        va, vb = a.get(key), b.get(key)
        if va is not None and vb is not None:
            out[key] = va
        else:
            v = va if va is not None else vb
            if v is not None:
                out[key] = v
    pa, pb = a.get("pos"), b.get("pos")
    out["pos"] = pa or pb or "A"
    return out


def _coord_np_merge_frozenset(fs1, fs2):
    d = _coord_np_merge_dict(dict_from_frozenset(fs1), dict_from_frozenset(fs2))
    return frozenset(d.items())


def _coord_ap_merge_frozenset(fs1, fs2):
    d = _coord_ap_merge_dict(dict_from_frozenset(fs1), dict_from_frozenset(fs2))
    return frozenset(d.items())


def _ip_subject_vp_agree(np_feats, vp_feats):
    """Подлежащее (NP) и вершина VP (финитный глагол): число; в прош. — род; в наст. — лицо."""
    if not _agree_if_both("number", np_feats, vp_feats):
        return False
    tense = vp_feats.get("tense")
    if tense == "past":
        # Род проверяем только при явном м/ж у подлежащего; нейтр. (мусье, дитя в разметке и т.д.)
        # часто не совпадает с окончанием глагола (по умолчанию м.р. в прошедшем).
        ng = np_feats.get("gender")
        if ng in ("m", "f") and not _agree_if_both("gender", np_feats, vp_feats):
            return False
    if tense == "pres":
        if not _agree_if_both("person", np_feats, vp_feats):
            return False
    return True


def _verb_np_direct_object_agree(verb_feats, np_feats):
    """
    NP как прямое дополнение справа от вершины глагола (финита или инфинитива):
    не именительный; при переходности — винительный.
    """
    if np_feats.get("case") == "nomn":
        return False
    if verb_feats.get("trans") == "tran":
        return np_feats.get("case") == "accs"
    return True


def _ip_vp_np_agree(vp_feats, np_feats):
    """
    IP -> VP NP (VP слева, NP справа).
    Если вершина VP — инфинитив, NP трактуем как дополнение инфинитива (иначе «ждать кошка»
    ошибочно проходит через согласование подлежащего с инфинитивом без лица/числа).
    Иначе — инверсия подлежащее–сказуемое для финита («пришёл он»).
    """
    if vp_feats.get("verb_form") == "infn":
        return _verb_np_direct_object_agree(vp_feats, np_feats)
    return _ip_subject_vp_agree(np_feats, vp_feats)


def _ip_predicative_np_ap(np_feats, ap_feats):
    """Безглагольное предложение с прилагательным сказуемым (опущена связка)."""
    if np_feats.get("case") is not None and np_feats.get("case") != "nomn":
        return False
    if ap_feats.get("case") is not None and ap_feats.get("case") != "nomn":
        return False
    return _np_modifier_head_agree(ap_feats, np_feats)


def _ip_predicative_np_np(subj_feats, pred_feats):
    """Безглагольное с именным сказуемым (Иван врач): обычно оба в И.п."""
    if subj_feats.get("case") is not None and subj_feats.get("case") != "nomn":
        return False
    if pred_feats.get("case") is not None and pred_feats.get("case") != "nomn":
        return False
    if not _agree_if_both("number", subj_feats, pred_feats):
        return False
    if not _agree_if_both("gender", subj_feats, pred_feats):
        return False
    return True


def _ip_np_pp_predicate(np_feats, _pp_feats):
    """Подлежащее + предложное сказуемое (он в Москве): подлежащее в И.п."""
    if np_feats.get("case") is not None and np_feats.get("case") != "nomn":
        return False
    return True


def _ip_pp_np_existential(_pp_feats, np_feats):
    """Экзистенциальное без «есть» (у меня кот): тема в И.п., если падеж задан."""
    if np_feats.get("case") is not None and np_feats.get("case") != "nomn":
        return False
    return True


def agreement_check(
    lhs,
    left_sym,
    left_fs,
    right_sym,
    right_fs,
    *,
    tokens=None,
    span_left=None,
    span_right=None,
):
    """
    Проверяет морфологическое согласование для конкретной конструкции.
    left_fs, right_fs - frozenset наборов (ключ, значение)
    Возвращает True, если комбинация допустима.
    tokens, span_left, span_right — для PP: отрезок [i, k) под предлог P.
    """
    lf = dict_from_frozenset(left_fs)
    rf = dict_from_frozenset(right_fs)
    # IP -> C IP: вводный союз/частица перед клаузой (А он ушёл)
    if lhs == "IP" and left_sym == "C" and right_sym == "IP":
        return True
    # IP -> IP ConjIP: бессоюзная / запятая / точка с запятой между предложениями
    if lhs == "IP" and left_sym == "IP" and right_sym == "ConjIP":
        return True
    if lhs == "ConjIP" and left_sym == "CommaC" and right_sym == "IP":
        return True
    # AP -> A N: пассивное причастие + твор. (пожираемый огнем) — падежи разные, не требуем совпадения
    if lhs == "AP" and left_sym == "A" and right_sym == "N":
        return True
    # AP -> AP NP: причастие + обстоятельство + твор. и т.п. (управляющего сразу всеми симфониями …)
    if lhs == "AP" and left_sym == "AP" and right_sym == "NP":
        return True
    # AP -> AP AP: определения подряд без запятой (большой красный …)
    if lhs == "AP" and left_sym == "AP" and right_sym == "AP":
        return _np_modifier_head_agree(lf, rf)
    # NP -> NP ConjCP: коррелят и относительные (предвкушение того, что …)
    if lhs == "NP" and left_sym == "NP" and right_sym == "ConjCP":
        return True
    # NP -> NP NP: однородные дополнения / подлежащие подряд без запятой (Маша Петя …)
    if lhs == "NP" and left_sym == "NP" and right_sym == "NP":
        return _np_coord_agree(lf, rf)
    # Бинаризация VP -> V NP NP → Z -> NP NP: два дополнения с разными падежами (дать мне урок),
    # не однородные ИГ — падежное согласование между ними не применяем.
    if _is_binarization_aux(lhs) and left_sym == "NP" and right_sym == "NP":
        return True
    # NP -> AP N: прилагательное с существительным (частичные граммемы pymorphy — только где оба заданы)
    if lhs == "NP" and left_sym == "AP" and right_sym == "N":
        return _np_modifier_head_agree(lf, rf)
    if lhs == "NP" and left_sym == "AP" and right_sym == "NP":
        return _np_modifier_head_agree(lf, rf)
    # NP -> N AP (постпозитивное прилагательное)
    if lhs == "NP" and left_sym == "N" and right_sym == "AP":
        if _np_modifier_head_agree(rf, lf):
            return True
        # коррелят «то»: предвкушение того — N в И.п. + A в Р.п. (pymorphy: того как ADJF gent)
        if lf.get("case") == "nomn" and rf.get("case") == "gent":
            return _agree_if_both("gender", lf, rf) and _agree_if_both("number", lf, rf)
        return False
    # NP -> Det N / N Det (детерминативы в словаре как ADJF с род/числ/пад)
    if lhs == "NP" and left_sym == "Det" and right_sym == "N":
        return _np_modifier_head_agree(lf, rf)
    if lhs == "NP" and left_sym == "N" and right_sym == "Det":
        return _np_modifier_head_agree(rf, lf)
    # NP -> N N: комплемент во 2-й группе в род.п. (симфониями возжигания)
    if lhs == "NP" and left_sym == "N" and right_sym == "N":
        return rf.get("case") == "gent"
    # NP -> N NP: имя + ИГ-комплемент в род.р. (симфониями возжигания и испепеления)
    if lhs == "NP" and left_sym == "N" and right_sym == "NP":
        return rf.get("case") == "gent"
    # NP -> N ConjAP: имя + запятая + причастный / определительный хвост (дирижера, управляющего …)
    if lhs == "NP" and left_sym == "N" and right_sym == "ConjAP":
        return True
    # NP -> NP ConjNP: однородные ИГ (и, а, но, …)
    if lhs == "NP" and left_sym == "NP" and right_sym == "ConjNP":
        return _np_coord_agree(lf, rf)
    # VP -> VP ConjVP: однородные сказуемые
    if lhs == "VP" and left_sym == "VP" and right_sym == "ConjVP":
        return _vp_coord_agree(lf, rf)
    # VP -> VP VP: однородные сказуемые без запятой (встал сел)
    if lhs == "VP" and left_sym == "VP" and right_sym == "VP":
        return _vp_coord_agree(lf, rf)
    # AP -> AP ConjAP: однородные определения (род/число/падеж при наличии)
    if lhs == "AP" and left_sym == "AP" and right_sym == "ConjAP":
        return _np_modifier_head_agree(lf, rf)
    # AdvP -> AdvP ConjAdvP: однородные обстоятельства (морф. ограничения слабые)
    if lhs == "AdvP" and left_sym == "AdvP" and right_sym == "ConjAdvP":
        return True
    # PP -> P NP: падеж именной группы по словарю управления предлога
    if lhs == "PP" and left_sym == "P" and right_sym == "NP":
        if tokens is None or span_left is None:
            return True
        i0, i1 = span_left
        prep_phrase = " ".join(tokens[i0:i1])
        return _pp_prep_np_agree(prep_phrase, rf)
    # VP -> V NP: справа дополнение; переходный глагол — прямое дополнение в вин.п.
    # (для VP -> NP V порядок подлежащее/дополнение не размечен — вин.п. не требуем)
    if lhs == "VP" and left_sym == "V" and right_sym == "NP":
        return _verb_np_direct_object_agree(lf, rf)
    # IP -> NP VP / VP NP: подлежащее и финитный глагол (вершина VP)
    if lhs == "IP" and left_sym == "NP" and right_sym == "VP":
        return _ip_subject_vp_agree(lf, rf)
    if lhs == "IP" and left_sym == "VP" and right_sym == "NP":
        return _ip_vp_np_agree(lf, rf)
    # IP -> NP AP / NP NP: безглагольное сказуемое (опущена связка)
    if lhs == "IP" and left_sym == "NP" and right_sym == "AP":
        return _ip_predicative_np_ap(lf, rf)
    if lhs == "IP" and left_sym == "NP" and right_sym == "NP":
        return _ip_predicative_np_np(lf, rf)
    # IP -> AP NP: инверсия прилагательного сказуемого (красивая она)
    if lhs == "IP" and left_sym == "AP" and right_sym == "NP":
        return _ip_predicative_np_ap(rf, lf)
    # IP -> NP PP: предложное сказуемое (он в Москве)
    if lhs == "IP" and left_sym == "NP" and right_sym == "PP":
        return _ip_np_pp_predicate(lf, rf)
    # IP -> PP NP: экзистенциальное без «есть» (у меня кот)
    if lhs == "IP" and left_sym == "PP" and right_sym == "NP":
        return _ip_pp_np_existential(lf, rf)
    # остальные случаи пропускаем
    return True

def merge_features(lhs, left_sym, left_fs, right_sym, right_fs):
    """
    Возвращает frozenset признаков для получившейся составляющей.
    Для NP вершина N, для VP вершина V и т.д.
    """
    lf = dict_from_frozenset(left_fs)
    rf = dict_from_frozenset(right_fs)
    if lhs == "ConjNP" and left_sym in ("C", "Comma") and right_sym == "NP":
        return right_fs
    if lhs == "ConjVP" and left_sym in ("C", "Comma") and right_sym == "VP":
        return right_fs
    if lhs == "ConjAP" and left_sym in ("C", "Comma") and right_sym == "AP":
        return right_fs
    if lhs == "ConjAdvP" and left_sym in ("C", "Comma") and right_sym == "AdvP":
        return right_fs
    if lhs == "ConjCP" and left_sym == "Comma" and right_sym == "CP":
        return right_fs
    if lhs == "ConjIP" and left_sym in ("Comma", "Semicolon", "Colon", "C") and right_sym == "IP":
        return right_fs
    if lhs == "ConjIP" and left_sym == "CommaC" and right_sym == "IP":
        return right_fs
    if lhs == "CommaC" and left_sym == "Comma" and right_sym == "C":
        return right_fs
    if lhs == "CP" and left_sym == "C" and right_sym == "IP":
        return right_fs
    if lhs == "NP":
        if left_sym == "NP" and right_sym == "ConjNP":
            return _coord_np_merge_frozenset(left_fs, right_fs)
        if left_sym == "NP" and right_sym == "NP":
            return _coord_np_merge_frozenset(left_fs, right_fs)
        if left_sym == "NP" and right_sym == "ConjCP":
            return left_fs
        # прилагательное + ИГ (символическим числом 451): голова — внутренняя ИГ
        if left_sym == "AP" and right_sym == "NP":
            return right_fs
        if left_sym == "N" and right_sym == "N":
            return left_fs
        if left_sym == "N" and right_sym == "NP":
            return left_fs
        # вершина N; если правило N что-то, берём от N
        if right_sym == "N": return right_fs
        if left_sym == "N": return left_fs
    if lhs == "VP":
        if left_sym == "VP" and right_sym == "ConjVP":
            return left_fs
        if left_sym == "VP" and right_sym == "VP":
            return left_fs
        # вершина глагол
        if left_sym == "V": return left_fs
        if right_sym == "V": return right_fs
    if lhs == "AP":
        if left_sym == "AP" and right_sym == "NP":
            return left_fs
        if left_sym == "AP" and right_sym == "ConjAP":
            return _coord_ap_merge_frozenset(left_fs, right_fs)
        if left_sym == "AP" and right_sym == "AP":
            return _coord_ap_merge_frozenset(left_fs, right_fs)
        if left_sym == "A" and right_sym == "N":
            return left_fs
        if 'A' in [left_sym, right_sym]:
            return left_fs if left_sym == 'A' else right_fs
    if _is_binarization_aux(lhs) and left_sym == "NP" and right_sym == "NP":
        return _coord_np_merge_frozenset(left_fs, right_fs)
    if lhs == "AdvP":
        if left_sym == "AdvP" and right_sym == "ConjAdvP":
            return left_fs
    if lhs == "IP":
        if left_sym == "C" and right_sym == "IP":
            return right_fs
        if left_sym == "IP" and right_sym == "ConjIP":
            return left_fs
        if left_sym == "NP" and right_sym in ("AP", "NP", "PP"):
            return left_fs
        if left_sym == "AP" and right_sym == "NP":
            return right_fs
        if left_sym == "PP" and right_sym == "NP":
            return right_fs
    # по умолчанию возвращаем пустой набор
    return frozenset()