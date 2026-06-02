def get_features(word):
	punct_feats = _get_punctuation_features(word)
	if punct_feats is not None:
		return punct_feats
		
	digit_feats = _get_digit_features(word)
	if digit_feats is not None:
		return digit_feats
	
	parses = morph.parse(word)
	filtered = [p for p in parses if p.score >= MIN_PARSE_SCORE]
	if not filtered:
		filtered = parses[:1]
	
	results = []
	for p in filtered:
		feats = {'pos': POS_MAP.get(p.tag.POS, 'X')}
		g = p.tag.grammemes
		
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
		
		verb_feats = _extract_verb_features(g, p.tag.POS)
		feats.update(verb_feats)
		
		results.append(feats)
		
	if len(results) == 1 and results[0].get("pos") == "X":
		core = word.replace("ё", "е").replace("Ё", "Е")
		if len(core) >= 2 and core[0].isupper() and core.replace("-", "").isalpha():
			return [{"pos": "N", "number": "sg", "case": "nomn"}]
	return results
