def extract_trees(i, j, symbol, word_chain, dp, grammar, memo, target_feat=None):
    feats_in_cell = dp[i][j].get(symbol, set())
    if target_feat is not None and target_feat not in feats_in_cell:
        return []

    memo_key = (i, j, symbol, target_feat)
    if memo_key in memo:
        return memo[memo_key]

    trees = []

    if j - i == 1:
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
        for lhs, rules in grammar.items():
            if lhs != symbol:
                continue
            for rhs in rules:
                if len(rhs) != 1:
                    continue
                child_sym = rhs[0]
                for fs in feats_in_cell:
                    if target_feat is not None and fs != target_feat:
                        continue
                    child_trees = extract_trees(i, j, child_sym, word_chain, dp, grammar, memo, target_feat=fs)
                    for child in child_trees:
                        feats_dict = dict_from_frozenset(fs)
                        trees.append({
                            'tag': symbol,
                            'feats': feats_dict,
                            'children': [child]
                        })
        memo[memo_key] = trees
        return trees

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
