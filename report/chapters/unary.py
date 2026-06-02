def _unary_closure_round(dp, unary_index, tokens, n):
    changed = False
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            for sym, feat_set in list(dp[i][j].items()): 
                for lhs in unary_index.get(sym, []):
                    for feat in feat_set:
                        if not agreement_check_unary(lhs, sym, feat):
                            continue
                        new_feat = merge_features_unary(lhs, sym, feat)
                        bucket = dp[i][j].setdefault(lhs, set())
                        before = len(bucket)
                        bucket.add(new_feat)
                        if len(bucket) > before:
                            changed = True
    return changed
