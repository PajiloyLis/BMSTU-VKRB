"""
Тесты для build_cyk_table из cyk.py.

Классы эквивалентности:
  КЭ-1  Пустой вход (n=0)
  КЭ-2  Одиночный токен, есть в индексах
  КЭ-3  Токен с несколькими морфоинтерпретациями (POS-омонимия)
  КЭ-4  Унарное замыкание обязательно (IP выводится только через цепочку унарных)
  КЭ-5  Корректное бинарное согласование → IP в dp[0][n]
  КЭ-6  Нарушенное согласование → IP отсутствует в dp[0][n]
  КЭ-7  Вложенная структура (NP внутри VP внутри IP): проверяем промежуточные ячейки
  КЭ-8  Токен отсутствует в обоих индексах: таблица строится, IP не возникает

Зависимости намеренно не импортируются из проекта — все индексы строятся
прямо в тестах, чтобы тесты были автономны и детерминированы.
"""

import pytest
from cyk import build_cyk_table


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------

def fs(**kwargs):
    """Сокращение: frozenset из именованных аргументов."""
    return frozenset(kwargs.items())


def cell(dp, i, j):
    """Возвращает словарь нетерминал→множество frozenset для ячейки dp[i][j]."""
    return dp[i][j]


def has_symbol(dp, i, j, sym):
    """Есть ли нетерминал sym в ячейке dp[i][j] с хотя бы одним набором признаков."""
    return bool(dp[i][j].get(sym))


# ---------------------------------------------------------------------------
# КЭ-1  Пустой вход
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """КЭ-1: token_feature_pairs = [] → функция возвращает таблицу 1×1 без ошибок."""

    def test_returns_without_error(self):
        dp = build_cyk_table([], {}, {})
        assert dp is not None

    def test_table_shape_is_1x1(self):
        dp = build_cyk_table([], {}, {})
        # n=0 → dp имеет размер (n+1)×(n+1) = 1×1
        assert len(dp) == 1
        assert len(dp[0]) == 1

    def test_single_cell_is_empty(self):
        dp = build_cyk_table([], {}, {})
        assert dp[0][0] == {}


# ---------------------------------------------------------------------------
# КЭ-2  Одиночный токен, есть в индексах
# ---------------------------------------------------------------------------

class TestSingleToken:
    """КЭ-2: n=1, токен с одним POS; унарный индекс даёт NP и IP."""

    @pytest.fixture
    def setup(self):
        # «кот»: N, sg, nomn
        noun_fs = fs(pos="N", number="sg", case="nomn", gender="m")
        pairs = [("кот", [{"pos": "N", "number": "sg", "case": "nomn", "gender": "m"}])]
        unary = {
            "N":  ["NP"],
            "NP": ["IP"],
        }
        binary = {}
        dp = build_cyk_table(pairs, unary, binary)
        return dp, noun_fs

    def test_terminal_pos_in_cell(self, setup):
        dp, _ = setup
        assert has_symbol(dp, 0, 1, "N")

    def test_unary_np_propagated(self, setup):
        dp, _ = setup
        assert has_symbol(dp, 0, 1, "NP")

    def test_unary_ip_propagated(self, setup):
        dp, _ = setup
        assert has_symbol(dp, 0, 1, "IP")

    def test_binary_loop_does_not_run(self, setup):
        # При n=1 диапазон length=2..1 пуст — ячейки i,j с j-i>1 не заполняются.
        dp, _ = setup
        # dp[0][1] существует (длина 1), dp[1][1] — диагональ (длина 0, пустая)
        assert dp[1][1] == {}


# ---------------------------------------------------------------------------
# КЭ-3  Токен с несколькими морфоинтерпретациями (POS-омонимия)
# ---------------------------------------------------------------------------

class TestMultipleInterpretations:
    """КЭ-3: один токен имеет несколько pos (омонимия N/V) — оба кладутся в ячейку."""

    @pytest.fixture
    def dp(self):
        # «стали»: N (gent/pl) ИЛИ V (past/pl)
        pairs = [
            ("стали", [
                {"pos": "N", "number": "sg", "case": "gent"},
                {"pos": "V", "number": "pl", "tense": "past", "verb_form": "fin"},
            ])
        ]
        dp = build_cyk_table(pairs, {}, {})
        return dp

    def test_noun_interpretation_present(self, dp):
        assert has_symbol(dp, 0, 1, "N")

    def test_verb_interpretation_present(self, dp):
        assert has_symbol(dp, 0, 1, "V")

    def test_noun_feats_correct(self, dp):
        n_feats = dp[0][1]["N"]
        assert fs(pos="N", number="sg", case="gent") in n_feats

    def test_verb_feats_correct(self, dp):
        v_feats = dp[0][1]["V"]
        assert fs(pos="V", number="pl", tense="past", verb_form="fin") in v_feats


# ---------------------------------------------------------------------------
# КЭ-4  Унарное замыкание обязательно
# ---------------------------------------------------------------------------

class TestUnaryClosure:
    """КЭ-4: IP недостижим за один проход унарных правил; нужна итерация замыкания.

    Цепочка: V → VP → IP (два шага).
    Без повторного прохода _unary_closure_round IP не появится.
    """

    @pytest.fixture
    def dp(self):
        pairs = [("бежит", [{"pos": "V", "number": "sg", "verb_form": "fin"}])]
        unary = {
            "V":  ["VP"],   # шаг 1
            "VP": ["IP"],   # шаг 2 (нужна итерация)
        }
        dp = build_cyk_table(pairs, unary, {})
        return dp

    def test_vp_reached(self, dp):
        assert has_symbol(dp, 0, 1, "VP")

    def test_ip_reached_after_closure(self, dp):
        assert has_symbol(dp, 0, 1, "IP")


# ---------------------------------------------------------------------------
# КЭ-5  Корректное бинарное согласование → IP в dp[0][n]
# ---------------------------------------------------------------------------

class TestBinaryAgreementValid:
    """КЭ-5: «кот спит» — число/лицо совпадают → IP строится.

    Минимальная грамматика: N→NP, V→VP, (NP,VP)→IP.
    """

    @pytest.fixture
    def dp(self):
        pairs = [
            ("кот",  [{"pos": "N", "number": "sg", "case": "nomn", "gender": "m"}]),
            ("спит", [{"pos": "V", "number": "sg", "person": "3",
                       "tense": "pres", "verb_form": "fin"}]),
        ]
        unary  = {"N": ["NP"], "V": ["VP"]}
        binary = {("NP", "VP"): ["IP"]}
        return build_cyk_table(pairs, unary, binary)

    def test_ip_in_root_cell(self, dp):
        assert has_symbol(dp, 0, 2, "IP")

    def test_np_in_left_cell(self, dp):
        assert has_symbol(dp, 0, 1, "NP")

    def test_vp_in_right_cell(self, dp):
        assert has_symbol(dp, 1, 2, "VP")


# ---------------------------------------------------------------------------
# КЭ-6  Нарушенное согласование → IP отсутствует
# ---------------------------------------------------------------------------

class TestBinaryAgreementInvalid:
    """КЭ-6: «кот спят» — число не совпадает → IP не строится, таблица без ошибок.

    agreement_check для IP→NP VP проверяет number; sg vs pl → False.
    """

    @pytest.fixture
    def dp(self):
        pairs = [
            ("кот",  [{"pos": "N", "number": "sg", "case": "nomn", "gender": "m"}]),
            ("спят", [{"pos": "V", "number": "pl", "person": "3",
                       "tense": "pres", "verb_form": "fin"}]),
        ]
        unary  = {"N": ["NP"], "V": ["VP"]}
        binary = {("NP", "VP"): ["IP"]}
        return build_cyk_table(pairs, unary, binary)

    def test_ip_absent_from_root(self, dp):
        assert not has_symbol(dp, 0, 2, "IP")

    def test_np_still_built(self, dp):
        assert has_symbol(dp, 0, 1, "NP")

    def test_vp_still_built(self, dp):
        assert has_symbol(dp, 1, 2, "VP")

    def test_no_exception_raised(self):
        """Таблица строится без исключений даже при несогласованности."""
        pairs = [
            ("кот",  [{"pos": "N", "number": "sg", "case": "nomn"}]),
            ("спят", [{"pos": "V", "number": "pl", "verb_form": "fin"}]),
        ]
        unary  = {"N": ["NP"], "V": ["VP"]}
        binary = {("NP", "VP"): ["IP"]}
        dp = build_cyk_table(pairs, unary, binary)  # не должно бросить
        assert dp is not None


# ---------------------------------------------------------------------------
# КЭ-7  Вложенная структура (NP внутри VP внутри IP)
# ---------------------------------------------------------------------------

class TestNestedStructure:
    """КЭ-7: «кошка ест рыбу» (SVO).

    Структура: IP → NP VP, VP → V NP.
    Проверяем, что промежуточная ячейка dp[1][3] содержит VP.
    """

    @pytest.fixture
    def dp(self):
        pairs = [
            ("кошка", [{"pos": "N", "number": "sg", "case": "nomn", "gender": "f"}]),
            ("ест",   [{"pos": "V", "number": "sg", "person": "3",
                        "tense": "pres", "verb_form": "fin", "trans": "tran"}]),
            ("рыбу",  [{"pos": "N", "number": "sg", "case": "accs", "gender": "f"}]),
        ]
        unary  = {"N": ["NP"], "V": ["VP"]}
        binary = {
            ("V",  "NP"): ["VP"],
            ("NP", "VP"): ["IP"],
        }
        return build_cyk_table(pairs, unary, binary)

    def test_subject_np_built(self, dp):
        assert has_symbol(dp, 0, 1, "NP")

    def test_object_np_built(self, dp):
        assert has_symbol(dp, 2, 3, "NP")

    def test_vp_spans_verb_and_object(self, dp):
        # VP должен покрывать «ест рыбу» (позиции 1–3)
        assert has_symbol(dp, 1, 3, "VP")

    def test_ip_in_root(self, dp):
        assert has_symbol(dp, 0, 3, "IP")

    def test_no_ip_in_verb_only_span(self, dp):
        # Один глагол «ест» без объекта не образует IP (только VP)
        assert has_symbol(dp, 1, 2, "VP")
        assert not has_symbol(dp, 1, 2, "IP")


# ---------------------------------------------------------------------------
# КЭ-8  Токен отсутствует в индексах
# ---------------------------------------------------------------------------

class TestTokenNotInIndexes:
    """КЭ-8: токен с pos='X' (неизвестное слово) не участвует ни в каких правилах.

    Таблица строится без ошибок; IP не возникает.
    """

    @pytest.fixture
    def dp(self):
        pairs = [("хрумзик", [{"pos": "X"}])]
        unary = {"N": ["NP"], "NP": ["IP"]}  # 'X' здесь нет
        binary = {("NP", "VP"): ["IP"]}
        return build_cyk_table(pairs, unary, binary)

    def test_no_exception(self):
        pairs = [("хрумзик", [{"pos": "X"}])]
        dp = build_cyk_table(pairs, {"N": ["NP"]}, {})
        assert dp is not None

    def test_only_pos_x_in_cell(self, dp):
        assert "X" in dp[0][1]

    def test_no_np_derived(self, dp):
        assert not has_symbol(dp, 0, 1, "NP")

    def test_no_ip_derived(self, dp):
        assert not has_symbol(dp, 0, 1, "IP")


# ---------------------------------------------------------------------------
# Вспомогательные функции cyk (прямой вызов)
# ---------------------------------------------------------------------------

class TestCykHelpers:
    def test_dict_from_frozenset(self):
        from cyk import dict_from_frozenset

        d = dict_from_frozenset(fs(pos="N", number="sg"))
        assert d == {"pos": "N", "number": "sg"}

    def test_agreement_check_unary_ip_np(self):
        from cyk import agreement_check_unary

        feat = fs(pos="N", number="sg", case="nomn")
        assert agreement_check_unary("IP", "NP", feat)
