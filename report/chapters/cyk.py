def build_cyk_table(token_feature_pairs, unary_index, binary_index):
    n = len(token_feature_pairs)
    tokens = [pair[0] for pair in token_feature_pairs]
    dp = [[dict() for _ in range(n+1)] for _ in range(n+1)]
    
    for j in range(1, n+1):
        token, features_list = token_feature_pairs[j-1]
        for feat in features_list:
            pos = feat['pos']
            feat_set = frozenset(feat.items())
            dp[j-1][j].setdefault(pos, set()).add(feat_set)
            for lhs in unary_index.get(pos, []):
                dp[j-1][j].setdefault(lhs, set()).add(feat_set)
    
    while _unary_closure_round(dp, unary_index, tokens, n):
        pass

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            for k in range(i + 1, j):
                left_cell = dp[i][k]
                right_cell = dp[k][j]
                for symA, featsA_set in left_cell.items():
                    for symB, featsB_set in right_cell.items():
                        for lhs in binary_index.get((symA, symB), []):
                            for featA in featsA_set:
                                for featB in featsB_set:
                                    if agreement_check(
                                        lhs, symA, featA, symB, featB,
                                        tokens=tokens,
                                        span_left=(i, k),
                                        span_right=(k, j),
                                    ):
                                        new_feat = merge_features(lhs, symA, featA, symB, featB)
                                        dp[i][j].setdefault(lhs, set()).add(new_feat)
        while _unary_closure_round(dp, unary_index, tokens, n):
            pass
    return dp
