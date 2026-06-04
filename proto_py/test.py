"""
Тесты для build_cyk_table и extract_trees.
Классы эквивалентности для build_cyk_table (КЭ-1 ... КЭ-8) и для extract_trees
(согласно таблице в технологическом разделе) — все реализованы.
"""

import pytest
from cyk import build_cyk_table
from tree_utils import extract_trees
from grammar import BINARIZATION_AUX_PREFIX


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------

def fs(**kwargs):
    return frozenset(kwargs.items())


def has_symbol(dp, i, j, sym):
    return bool(dp[i][j].get(sym))


# ---------------------------------------------------------------------------
# Тесты для build_cyk_table (КЭ-1 ... КЭ-8) — без изменений
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_returns_without_error(self):
        dp = build_cyk_table([], {}, {})
        assert dp is not None
    def test_table_shape_is_1x1(self):
        dp = build_cyk_table([], {}, {})
        assert len(dp) == 1 and len(dp[0]) == 1
    def test_single_cell_is_empty(self):
        dp = build_cyk_table([], {}, {})
        assert dp[0][0] == {}


class TestSingleToken:
    @pytest.fixture
    def setup(self):
        pairs = [("кот", [{"pos": "N", "number": "sg", "case": "nomn", "gender": "m"}])]
        unary = {"N": ["NP"], "NP": ["IP"]}
        dp = build_cyk_table(pairs, unary, {})
        return dp
    def test_terminal_pos_in_cell(self, setup):
        assert has_symbol(setup, 0, 1, "N")
    def test_unary_np_propagated(self, setup):
        assert has_symbol(setup, 0, 1, "NP")
    def test_unary_ip_propagated(self, setup):
        assert has_symbol(setup, 0, 1, "IP")


class TestMultipleInterpretations:
    @pytest.fixture
    def dp(self):
        pairs = [("стали", [
            {"pos": "N", "number": "sg", "case": "gent"},
            {"pos": "V", "number": "pl", "tense": "past", "verb_form": "fin"},
        ])]
        return build_cyk_table(pairs, {}, {})
    def test_noun_interpretation_present(self, dp):
        assert has_symbol(dp, 0, 1, "N")
    def test_verb_interpretation_present(self, dp):
        assert has_symbol(dp, 0, 1, "V")


class TestUnaryClosure:
    @pytest.fixture
    def dp(self):
        pairs = [("бежит", [{"pos": "V", "number": "sg", "verb_form": "fin"}])]
        unary = {"V": ["VP"], "VP": ["IP"]}
        return build_cyk_table(pairs, unary, {})
    def test_vp_reached(self, dp):
        assert has_symbol(dp, 0, 1, "VP")
    def test_ip_reached_after_closure(self, dp):
        assert has_symbol(dp, 0, 1, "IP")


class TestBinaryAgreementValid:
    @pytest.fixture
    def dp(self):
        pairs = [
            ("кот",  [{"pos": "N", "number": "sg", "case": "nomn", "gender": "m"}]),
            ("спит", [{"pos": "V", "number": "sg", "person": "3", "tense": "pres", "verb_form": "fin"}])
        ]
        unary = {"N": ["NP"], "V": ["VP"]}
        binary = {("NP", "VP"): ["IP"]}
        return build_cyk_table(pairs, unary, binary)
    def test_ip_in_root_cell(self, dp):
        assert has_symbol(dp, 0, 2, "IP")
    def test_np_in_left_cell(self, dp):
        assert has_symbol(dp, 0, 1, "NP")
    def test_vp_in_right_cell(self, dp):
        assert has_symbol(dp, 1, 2, "VP")


class TestBinaryAgreementInvalid:
    @pytest.fixture
    def dp(self):
        pairs = [
            ("кот",  [{"pos": "N", "number": "sg", "case": "nomn", "gender": "m"}]),
            ("спят", [{"pos": "V", "number": "pl", "person": "3", "tense": "pres", "verb_form": "fin"}])
        ]
        unary = {"N": ["NP"], "V": ["VP"]}
        binary = {("NP", "VP"): ["IP"]}
        return build_cyk_table(pairs, unary, binary)
    def test_ip_absent_from_root(self, dp):
        assert not has_symbol(dp, 0, 2, "IP")
    def test_np_still_built(self, dp):
        assert has_symbol(dp, 0, 1, "NP")
    def test_vp_still_built(self, dp):
        assert has_symbol(dp, 1, 2, "VP")


class TestNestedStructure:
    @pytest.fixture
    def dp(self):
        pairs = [
            ("кошка", [{"pos": "N", "number": "sg", "case": "nomn", "gender": "f"}]),
            ("ест",   [{"pos": "V", "number": "sg", "person": "3", "tense": "pres", "verb_form": "fin", "trans": "tran"}]),
            ("рыбу",  [{"pos": "N", "number": "sg", "case": "accs", "gender": "f"}])
        ]
        unary = {"N": ["NP"], "V": ["VP"]}
        binary = {("V", "NP"): ["VP"], ("NP", "VP"): ["IP"]}
        return build_cyk_table(pairs, unary, binary)
    def test_subject_np_built(self, dp):
        assert has_symbol(dp, 0, 1, "NP")
    def test_object_np_built(self, dp):
        assert has_symbol(dp, 2, 3, "NP")
    def test_vp_spans_verb_and_object(self, dp):
        assert has_symbol(dp, 1, 3, "VP")
    def test_ip_in_root(self, dp):
        assert has_symbol(dp, 0, 3, "IP")
    def test_no_ip_in_verb_only_span(self, dp):
        assert has_symbol(dp, 1, 2, "VP")
        assert not has_symbol(dp, 1, 2, "IP")


class TestTokenNotInIndexes:
    @pytest.fixture
    def dp(self):
        pairs = [("хрумзик", [{"pos": "X"}])]
        unary = {"N": ["NP"], "NP": ["IP"]}
        return build_cyk_table(pairs, unary, {})
    def test_only_pos_x_in_cell(self, dp):
        assert "X" in dp[0][1]
    def test_no_np_derived(self, dp):
        assert not has_symbol(dp, 0, 1, "NP")
    def test_no_ip_derived(self, dp):
        assert not has_symbol(dp, 0, 1, "IP")


# ---------------------------------------------------------------------------
# Тесты для extract_trees (исправленные)
# ---------------------------------------------------------------------------

class TestExtractTrees:

    # КЭ-ET-1: Терминал на отрезке длины 1
    def test_terminal_length1(self):
        dp = [[dict() for _ in range(2)] for _ in range(2)]
        feat_set = {fs(pos="N", number="sg")}
        dp[0][1] = {"N": feat_set}
        tokens = ["кот"]
        grammar = {}
        result = extract_trees(0, 1, "N", tokens, dp, grammar, {})
        assert len(result) == 1
        node = result[0]
        assert node['tag'] == "N"
        assert node['word'] == "кот"
        assert node['feats'] == {"pos": "N", "number": "sg"}

    # КЭ-ET-2: Нетерминал на отрезке длины 1, выведенный через унарное правило
    def test_unary_length1(self):
        dp = [[dict() for _ in range(2)] for _ in range(2)]
        # Признаки для N и NP должны быть одинаковыми (копируются при унарном выводе)
        common_feat = {fs(pos="N", number="sg")}
        dp[0][1] = {"N": common_feat, "NP": common_feat}   # NP получило те же признаки
        tokens = ["кот"]
        grammar = {"NP": [["N"]]}
        result = extract_trees(0, 1, "NP", tokens, dp, grammar, {})
        assert len(result) == 1
        node = result[0]
        assert node['tag'] == "NP"
        assert len(node['children']) == 1
        child = node['children'][0]
        assert child['tag'] == "N"
        assert child['word'] == "кот"
        assert child['feats'] == {"pos": "N", "number": "sg"}

    # КЭ-ET-3: Соответствующий символ отсутствует в ячейке
    def test_symbol_missing(self):
        dp = [[dict() for _ in range(2)] for _ in range(2)]
        dp[0][1] = {"N": {fs(number="sg")}}
        tokens = ["кот"]
        grammar = {}
        result = extract_trees(0, 1, "VP", tokens, dp, grammar, {})
        assert result == []

    # КЭ-ET-4: Символ в ячейке не согласуется с target_feat
    def test_target_feat_mismatch(self):
        dp = [[dict() for _ in range(2)] for _ in range(2)]
        feat_set = {fs(pos="N", number="sg", case="nomn")}
        dp[0][1] = {"N": feat_set}
        tokens = ["кот"]
        grammar = {}
        target = fs(pos="V", number="sg", tense="past")
        result = extract_trees(0, 1, "N", tokens, dp, grammar, {}, target_feat=target)
        assert result == []

    # КЭ-ET-5: Токен с несколькими наборами морфологических признаков
    def test_multiple_feats_terminal(self):
        dp = [[dict() for _ in range(2)] for _ in range(2)]
        feats_set = {
            fs(pos="N", number="sg", case="gent"),
            fs(pos="V", number="pl", tense="past", verb_form="fin"),
        }
        dp[0][1] = {"N": feats_set, "V": feats_set}
        tokens = ["стали"]
        grammar = {}
        result = extract_trees(0, 1, "N", tokens, dp, grammar, {})
        assert len(result) == 2
        feats_list = [tuple(sorted(r['feats'].items())) for r in result]
        assert len(set(feats_list)) == 2

    # КЭ-ET-6: Нетерминал на отрезке длины >1, выведенный через бинарное правило
    def test_binary_rule(self):
        n = 2
        dp = [[dict() for _ in range(n+1)] for _ in range(n+1)]
        v_feat = {fs(pos="V", number="sg", tense="pres")}
        n_feat = {fs(pos="N", number="sg", case="accs")}
        dp[0][1] = {"V": v_feat}
        dp[1][2] = {"N": n_feat}
        dp[1][2]["NP"] = n_feat            # NP ← N
        dp[0][2] = {"VP": v_feat}          # VP ← V NP
        tokens = ["ест", "рыбу"]
        grammar = {"VP": [["V", "NP"]], "NP": [["N"]]}
        result = extract_trees(0, 2, "VP", tokens, dp, grammar, {})
        assert len(result) == 1

    # КЭ-ET-7: Нетерминал на отрезке длины >1, выведенный через унарное правило
    def test_unary_on_longer_span(self):
        n = 2
        dp = [[dict() for _ in range(n+1)] for _ in range(n+1)]
        n_feat = {fs(pos="N", number="sg", case="nomn")}
        v_feat = {fs(pos="V", number="sg", tense="pres")}
        dp[0][1] = {"N": n_feat}
        dp[1][2] = {"V": v_feat}
        dp[0][1]["NP"] = n_feat
        dp[1][2]["VP"] = v_feat
        dp[0][2] = {"VP": v_feat}
        dp[0][2]["IP"] = v_feat
        tokens = ["кот", "спит"]
        grammar = {"IP": [["VP"]], "VP": [["NP", "V"]], "NP": [["N"]]}
        result = extract_trees(0, 2, "IP", tokens, dp, grammar, {})
        assert len(result) == 1

    # КЭ-ET-8: Мемоизация
    def test_memoization(self):
        dp = [[dict() for _ in range(2)] for _ in range(2)]
        feat_set = {fs(pos="N")}
        dp[0][1] = {"N": feat_set}
        tokens = ["слово"]
        grammar = {}
        memo = {}
        first = extract_trees(0, 1, "N", tokens, dp, grammar, memo)
        second = extract_trees(0, 1, "N", tokens, dp, grammar, memo)
        assert len(first) == len(second)
        assert memo[(0, 1, "N", None)] is first

    # КЭ-ET-9: Вспомогательный нетерминал бинаризации __AUX_i
    def test_aux_binarization_tag(self):
        n = 3
        dp = [[dict() for _ in range(n+1)] for _ in range(n+1)]
        v_feat = {fs(pos="V")}
        n_feat = {fs(pos="N")}
        p_feat = {fs(pos="P")}
        dp[0][1] = {"V": v_feat}
        dp[1][2] = {"N": n_feat}
        dp[2][3] = {"P": p_feat}
        dp[1][2]["NP"] = n_feat
        dp[2][3]["PP"] = p_feat
        dp[1][3] = {"__AUX_0": {fs()}}
        dp[0][3] = {"VP": v_feat}
        tokens = ["любит", "кошек", "очень"]
        grammar = {"VP": [["V", "__AUX_0"]], "__AUX_0": [["NP", "PP"]], "NP": [["N"]], "PP": [["P"]]}
        result = extract_trees(1, 3, "__AUX_0", tokens, dp, grammar, {})   # исправлен i=1
        assert len(result) == 1
        node = result[0]
        assert node['tag'] == "__AUX_0"
        assert len(node['children']) == 2
        assert node['children'][0]['tag'] == "NP"
        assert node['children'][1]['tag'] == "PP"


if __name__ == "__main__":
    pytest.main(["--cov=cyk", "--cov=tree_utils", "--cov-report=term", "--cov-report=html", __file__])