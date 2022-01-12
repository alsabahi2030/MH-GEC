from nltk.stem.lancaster import LancasterStemmer
from rule_script import align_text as align_text
from rule_script import cat_rules as cat_rules
from rule_script import toolbox2 as toolbox
from utils import get_operation as get_operation

PLURAL_OPERATIONS = ['$SUFFIXTRANSFORM_VERB_VB_VBZ','$TRANSFORM_VERB_VB_VBZ','$SUFFIXTRANSFORM_Y_TO_IES','$TRANSFORM_AGREEMENT_PLURAL','$SUFFIXTRANSFORM_APPEND_es']
SINGULAR_OPERATIONS = ['$TRANSFORM_VERB_VBZ_VB','$SUFFIXTRANSFORM_IES_TO_Y','$TRANSFORM_AGREEMENT_SINGULAR']
VERB_TAG_MAP = {"VBD_VB":"VBD_VB","VBN_VB":"","VBD_VBN":"","VBN_VBD":"","VBD_VBG":"","VBN_VBG":"","VBD_VBZ":"","VBN_VBZ":"","VB_VBD":"","VB_VBN":"","VBG_VBD":"","VBG_VBN":"","VBZ_VBD":"","VBZ_VBN":""}
MAKE = ['made', 'make', 'making']
COORD_CONJ = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so']
SUBORD_CONJ = ['after', ' although', 'as', 'as if', 'as long as', 'as much as', 'as soon as', 'as though', 'because', 'before', 'even', 'even if', 'even though', 'if', 'if only', 'if when', 'if then', 'inasmuch', 'in order that', 'just as', 'lest', 'now', 'now since', 'now that', 'now when', 'once', 'provided', 'provided that', 'rather than', 'since', 'so that', 'supposing', 'than', 'that', 'though', 'til', 'unless', 'until', 'when', 'whenever', 'where', 'whereas', 'where if', 'wherever', 'whether', 'which', 'while', 'who', 'whoever', 'why']
CORRCONJ_1 = ['either', 'not only', 'neither', 'both', 'whether', 'just as', 'as', 'as much', 'no sooner', 'rather', 'not', 'such', 'scarcely', 'no sooner', 'rather']
CORRCONJ_2 = ['or', 'but (also)', 'nor', 'and', 'or', 'so', 'as', 'as', 'than', 'than', 'but', 'that', 'when ', 'than', 'than']
DO = ['do', 'did', 'doing']
INTERJ = ['ah', 'aha', 'ahem', 'alas', 'amen', 'aw', 'awesome', 'aww', 'bada-bing', 'bah', 'baloney', 'big deal', 'bingo', 'boo', 'boo-hoo', 'booyah', 'boo-yah', 'boy', 'boy oh boy', 'bravo', 'brilliant', 'brrr', 'bull', 'bye', 'bye-bye', 'cheers', 'yes', 'no', 'come on', 'cool', 'cowabunga', 'dang', 'darn', 'darn it', 'dear me', 'duck', 'duh', 'eh', 'enjoy', 'excellent', 'fabulous', 'fantastic', 'fiddledeedee', 'fiddle-dee-dee', 'finally', "for heaven's sake", 'fore', 'foul', 'freeze', 'gee', 'gee whiz', 'gee willikers', 'giddyap', 'giddyup', 'golly', 'good golly', 'golly gee willikers', 'goodbye', 'good-bye', 'good grief', 'good heavens', 'gosh', 'great', 'great balls of fire', 'ha', 'hallelujah', 'heavens', 'heavens above', 'heavens to betsy', 'heigh-ho', 'hello', 'help', 'hey', 'hey there', 'hi', 'hiya', 'hip', 'hip', 'hooray', 'hmm', 'hrm', 'ho-ho-ho', 'ho-hum', 'hooray', 'hurrah', 'hurray', 'howdy', 'howdy do', 'huh', 'ick', 'indeed', 'jeez', 'kaboom', 'kapow', 'lordy', 'lordy', 'lordy', 'mama mia', 'man', 'marvelous', 'my', 'my goodness', 'my heavens', 'my stars', 'my word', 'nah', 'no problem', 'no way', 'nope', 'nuts', 'oh', 'oh boy', 'oh dear', 'oh my', 'oh my gosh', 'oh my goodness', 'oh no', 'oh well', 'ok', 'okay', 'ouch', 'ow', 'please', 'poof', 'shh', 'super', 'swell', 'welcome', 'well', 'whoop-de-doo', 'woo-hoo', 'wow', 'yabba dabba doo', 'yadda, yadda, yadda', 'yippee', 'yummy']
ARTICLES = ['a', 'an','the']
PRONOUNS=['me','I','my']
PREPOSITIONS = [
    '', 'of', 'with', 'at', 'from', 'into', 'during', 'including', 'until', 'against', 'among', 'throughout',
    'despite', 'towards', 'upon', 'concerning', 'to', 'in', 'for', 'on', 'by', 'about', 'like',
    'through', 'over', 'before', 'between', 'after', 'since', 'without', 'under', 'within', 'along',
    'following', 'across', 'behind', 'beyond', 'plus', 'except', 'but', 'up', 'out', 'around', 'down'
    'off', 'above', 'near']
CONFUSED_WORDS = {'lose': 'loose', 'farther': 'further', 'bear': 'bare', 'compliment': 'complement', 'affect': 'effect', 'advice': 'advise', 'resign': 're-sign', 'breath': 'breathe', 'capital': 'capitol', 'empathy': 'sympathy', 'principal': 'principle', 'stationary': 'stationery', 'inquiry': 'enquiry', 'lay': 'lie', 'imply': 'infer', 'defence': 'defense', 'assure': 'ensure', 'accept': 'except', 'accurate': 'precise', 'adverse': 'averse', 'appraise': 'apprise', 'birth': 'berth', 'borrow': 'lend', 'cash': 'cache', 'comprise': 'compose', 'desert': 'dessert', 'several': 'various'}

def merge_gedits(edits):
	if edits:
		return [("X", edits[0][1], edits[-1][2], edits[0][3], edits[-1][4])]
	else:
		return edits

def format_errordetails_dictedits(start1, end1, errtype, oper, errorcat, ori_tok, cor_tok, ori_tokg,cor_tokg, start2, end2,detailed_explanatory,context_based_dict, general_error_exp_dict, lang='en'):
    if detailed_explanatory:
        if errorcat in context_based_dict.keys():
            if lang == 'en':
                contextbasedexplan = context_based_dict[errorcat][0]
                contextbasedexplan = contextbasedexplan.replace('<ori_tok>', f"\"{ori_tok}\"")
                contextbasedexplan = contextbasedexplan.replace('<cor_tok>', f"\"{cor_tok}\"")
                contextbasedexplan = contextbasedexplan.replace('<ori_tokg>', f"\"{ori_tokg}\"")
                contextbasedexplan = contextbasedexplan.replace('<cor_tokg>', f"\"{cor_tokg}\"")
            elif lang == 'cn':
                contextbasedexplan = context_based_dict[errorcat][1]
                contextbasedexplan = contextbasedexplan.replace('<ori_tok>', f"\"{ori_tok}\"")
                contextbasedexplan = contextbasedexplan.replace('<cor_tok>', f"\"{cor_tok}\"")
                contextbasedexplan = contextbasedexplan.replace('<ori_tokg>', f"\"{ori_tokg}\"")
                contextbasedexplan = contextbasedexplan.replace('<cor_tokg>', f"\"{cor_tokg}\"")
        else:
            if errtype.startswith('M:'):
                contextbasedexplan = f'There is a missing word in this context. Consider adding the word {cor_tok}.'
            elif errtype.startswith('R:'):
                contextbasedexplan = f'the word "{ori_tok}" does not seem to be correct in this context. Consider replacing it with "{cor_tok}".'
            elif errtype.startswith('U:'):
                contextbasedexplan = f'There is an unneccessary "{ori_tok}". Consider deleting it.'
            else:
                contextbasedexplan = f'Unknown error!'

        if errorcat in general_error_exp_dict.keys():
            errorcat = errorcat
        else:
            if errtype.startswith('M:'):
                #errorcat ='M:WORD'
                return {'fromx': start1, 'tox': end1, 'errortype': errtype, 'operation': oper,
                        'error category': errorcat,
                        'incorrect': ori_tok, 'correction': cor_tok, 'fromy': start2, 'toy': end2,
                        'context based explanation': contextbasedexplan,
                        'general explanation': "Others ... "}
            elif errtype.startswith('R:'):
                if ori_tok in CONFUSED_WORDS.keys() and CONFUSED_WORDS[ori_tok] == cor_tok:
                    errorcat = f"{ori_tok.lower()}vs{cor_tok.lower()}"
                elif cor_tok in CONFUSED_WORDS.keys() and CONFUSED_WORDS[cor_tok] == ori_tok:
                    errorcat = f"{cor_tok.lower()}vs{ori_tok.lower()}"
                elif ori_tok.lower() in ['me', 'my'] and cor_tok.lower() in ['me', 'my']:
                    errorcat = 'MEVSMY'
                elif ori_tok.lower() in ['me', 'i'] and cor_tok.lower() in ['me', 'i']:
                    errorcat = 'IVSME'
                elif ori_tok.lower != cor_tok.lower() and ori_tok.lower() in ['less', 'fewer'] and cor_tok.lower() in [
                    'less', 'fewer']:
                    errorcat = 'R:LESSVSFEWER'
                elif (cor_tok in DO and ori_tok in MAKE) or (
                        ori_tok in DO and cor_tok in MAKE):
                    errorcat = 'COMFUSEDOMAKE'
                else:
                    #errorcat ='WORDUSAGE'
                    return {'fromx': start1, 'tox': end1, 'errortype': errtype, 'operation': oper,
                            'error category': errorcat,
                            'incorrect': ori_tok, 'correction': cor_tok, 'fromy': start2, 'toy': end2,
                            'context based explanation': contextbasedexplan,
                            'general explanation': "Others ... "}
            elif errtype.startswith('U:'):
                #errorcat ='U:WORD'
                return {'fromx': start1, 'tox': end1, 'errortype': errtype, 'operation': oper,
                        'error category': errorcat,
                        'incorrect': ori_tok, 'correction': cor_tok, 'fromy': start2, 'toy': end2,
                        'context based explanation': contextbasedexplan,
                        'general explanation': "Others ... "}
            else:
                return {'fromx': start1, 'tox': end1, 'errortype': errtype, 'operation': oper,
                        'error category': errorcat,
                        'incorrect': ori_tok, 'correction': cor_tok, 'fromy': start2, 'toy': end2,
                        'context based explanation': contextbasedexplan,
                        'general explanation': "Others ... "}
        if errorcat == 'Others':
            return {'fromx': start1, 'tox': end1, 'errortype': errtype, 'operation': oper,
                    'error category': errorcat,
                    'incorrect': ori_tok, 'correction': cor_tok, 'fromy': start2, 'toy': end2,
                    'context based explanation': contextbasedexplan,
                    'general explanation': "Others ... "}
        return {'fromx': start1, 'tox': end1, 'errortype': errtype, 'operation': oper, 'error category': errorcat,
                    'incorrect': ori_tok, 'correction': cor_tok, 'fromy': start2, 'toy': end2,
                    'context based explanation': contextbasedexplan,'general explanation':general_error_exp_dict[errorcat]}
    else:
        return {'fromx':start1, 'tox': end1, 'errortype':errtype, 'operation': oper, 'error category': errorcat, 'incorrect': ori_tok,'correction':cor_tok,'fromy':start2,'toy':end2, 'context based explanation': contextbasedexplan}

def check_missing_gect(errtype,ori_tok,cor_tok,ori_tokg,oper):
    if errtype == 'M:PUNCT':
        if cor_tok in [',','!']:
            if ori_tokg in INTERJ:
                errorcat = 'M:INTERJ'
            elif cor_tok ==',':
                errorcat = 'Mcomma'
            else:
                errorcat = 'M:EXCL'
        elif cor_tok == '.':
            errorcat = 'M:PERIOD'
        else:
            errorcat = errtype
    elif errtype.startswith('M:VERB'):
        if errtype == 'M:VERB:INFIN':
            errorcat = errtype
        else:
            errorcat = 'M:VERB'
    elif errtype == 'M:ADJ':
        errorcat = errtype
    elif errtype == 'M:ADV':
        errorcat = errtype
    elif errtype == 'M:DET':
        if cor_tok.lower() in ['a', 'an', 'the']:
            errorcat = 'M:ART'
        else:
            errorcat = 'M:DET'
    elif errtype.startswith('M:NOUN'):
        if errtype == 'M:NOUN:POSS':
            errorcat = errtype
        else:
            errorcat = errtype
    elif errtype == 'M:PART':
        errorcat = errtype
    elif errtype == 'M:PREP':
        errorcat = errtype
    elif errtype == 'M:PRON':
        errorcat = 'PRON'
    elif errtype == "M:CONJ":
        if (cor_tok.lower() in COORD_CONJ):
            errorcat = "COORDCONJ"
        elif (cor_tok.lower() in SUBORD_CONJ):
            errorcat = "SUBORDCONJ"
        elif (cor_tok.lower() in CORRCONJ_1) or (cor_tok in CORRCONJ_2):
            errorcat = "CORRCONJ"
        else:
            errorcat = errtype
    elif cor_tok in PREPOSITIONS:
        errorcat = errtype
    else:
        errorcat = errtype
    return errorcat

def check_missing_rules(errtype,ori_tok,cor_tok):
    if errtype == 'M:PUNCT':
        if cor_tok == ',':
            errorcat = 'Mcomma'
        elif cor_tok == '.':
            errorcat = 'M:PERIOD'
        else:
            errorcat = errtype
    elif errtype.startswith('M:VERB'):
        if errtype == 'M:VERB:INFIN':
            errorcat = errtype
        else:
            errorcat = 'M:VERB'
    elif errtype == 'M:ADJ':
        errorcat = errtype
    elif errtype == 'M:ADV':
        errorcat = errtype
    elif errtype == 'M:DET':
        if cor_tok.lower() in ['a', 'an', 'the']:
            errorcat = 'M:ART'
        else:
            errorcat = 'M:DET'
    elif errtype.startswith('M:NOUN'):
        if errtype == 'M:NOUN:POSS':
            errorcat = errtype
        else:
            errorcat = errtype
    elif errtype == 'M:PART':
        errorcat = errtype
    elif errtype == 'M:PREP':
        errorcat = errtype
    elif errtype == 'M:PRON':
        errorcat = 'PRON'
    elif errtype == "M:CONJ":
        if (cor_tok.lower() in COORD_CONJ):
            errorcat = "COORDCONJ"
        elif (cor_tok.lower() in SUBORD_CONJ):
            errorcat = "SUBORDCONJ"
        elif (cor_tok.lower() in CORRCONJ_1) or (cor_tok in CORRCONJ_2):
            errorcat = "CORRCONJ"
        else:
            errorcat = 'CONJ'
    elif cor_tok in PREPOSITIONS:
        errorcat = errtype
    else:
        errorcat = errtype
    return errorcat

def check_replacing_gect(errtype,ori_tok,cor_tok,ori_tokg,ori_pos, cor_pos, oper,start1,uncountable_nouns, prev_ori_tok, prev_cor_tok, ori_dpos, cor_dpos):
    if errtype == 'R:OTHER':
        if ori_tok in CONFUSED_WORDS.keys() and cor_tok in CONFUSED_WORDS.values():
            errorcat=f"{ori_tok.lower()}vs{cor_tok.lower()}"
        elif ori_tok in CONFUSED_WORDS.values() and cor_tok in CONFUSED_WORDS.keys():
            errorcat=f"{cor_tok.lower()}vs{ori_tok.lower()}"
        elif ori_tok.lower() in ['me','my'] and cor_tok.lower() in ['me','my']:
            errorcat = 'MEVSMY'
        elif ori_tok.lower() in ['me','i'] and cor_tok.lower() in ['me','i']:
            errorcat = 'IVSMY'
        elif (ori_pos=='ADJ' and cor_pos == 'ADV') or (ori_pos=='ADV' and cor_pos == 'ADJ'):
            errorcat = 'ADJVSADV'
        elif oper in PLURAL_OPERATIONS:
            errorcat = 'SingularAfterPluralQuant'
        elif ori_tok.lower != cor_tok.lower() and ori_tok.lower() in ['less','fewer'] and cor_tok.lower() in  ['less','fewer']:
            errorcat = 'R:LESSVSFEWER'
        elif ori_tok[0] in [',','.','!','?'] or cor_tok[0] in [',','.','!','?'] and ori_tok[0] != cor_tok[0]:
            ori_tok1=ori_tok[0]
            #ori_tok2=ori_tok[1:].strip()
            cor_tok1=cor_tok[0]
            #cor_tok2=cor_tok[1:].strip()
            if ori_tok[0] == ',' and cor_tok[0] == '.':
                errorcat = 'COMSPLICE'
            elif ori_tok[0] == '.' and cor_tok[0] == '?':
                errorcat = 'PERIODINSQUEST'
            else:
                errorcat = 'R:PUNCT'
        else:
            errorcat = errtype
    elif errtype == 'R:PREP':
        errorcat = errtype
    elif 'R:NOUN' in errtype:
        if errtype == 'R:NOUN:POSS':
            errorcat = errtype
        elif errtype == 'R:NOUN:NUM':
            if oper in SINGULAR_OPERATIONS:
                if cor_tok in uncountable_nouns:
                    errorcat = 'UCPLURAL'
                else:
                    errorcat = 'INCPLUR'
            elif oper in PLURAL_OPERATIONS:
                errorcat = 'SingularAfterPluralQuant'
            else:
                errorcat = errtype
        elif oper == '$TRANSFORM_AGREEMENT_SINGULAR' and errtype == 'R:NOUN:POSS':
            errorcat = errtype
        else:
            errorcat = errtype
    elif errtype == 'R:SPELL':
        errorcat = errtype
    elif 'R:VERB' in errtype:
        if 'VB' in ori_dpos[0] and 'VB' in cor_dpos[0] and len(ori_dpos) == 1 and len(cor_dpos) == 1 and ori_dpos!=cor_dpos:
            errorcat = f'VERB_{ori_dpos[0]}_{cor_dpos[0]}'
        elif '$TRANSFORM_VERB' in oper:
            errorcat = oper.replace('$TRANSFORM_', '')
        elif '$SUFFIXTRANSFORM' in oper:
            if oper == '$SUFFIXTRANSFORM_ING_TO_E' or oper =="$SUFFIXTRANSFORM_REMOVE_ing":
                errorcat = 'VERB_VBG_VB'
            elif oper =='$SUFFIXTRANSFORM_APPEND_ing' or 'S_TO_ING' in oper or 'E_TO_ING' in oper or 'ES_TO_ING' in oper:
                errorcat = 'VERB_VB_VBG'
            elif oper =='$SUFFIXTRANSFORM_APPEND_ing' or 'S_TO_ING' in oper or 'ES_TO_ING' in oper:
                errorcat = 'VERB_VB_VBG'
            else:
                errorcat=errtype

        elif errtype == 'R:VERB:SVA':
            errorcat = errtype
        elif (cor_tok in DO and ori_tok in MAKE) or (
                ori_tok in DO and cor_tok in MAKE):
            errorcat = 'COMFUSEDOMAKE'
        elif errtype == 'R:VERB:INFL':
            if oper == '$SUFFIXTRANSFORM_REMOVE_d':
                errorcat = 'VERB_VBD_VB'
            else:
                errorcat = 'R:INFL'
        else:
            errorcat = errtype
    elif 'R:DET' in errtype:
        errorcat = 'IncoArt'
    elif errtype.startswith('R:ORTH'):
        errorcat = errtype
        if errtype == 'R:ORTH:Capt1st':
            if start1 == 0:
                errorcat = 'R:ORTH:Capt1stWord'
            elif ori_tok == 'i':
                errorcat = 'LowPronI'
        elif errtype == 'R:ORTH:HYPHEN':
            errorcat = 'HYPHEN'
        elif errtype == 'R:ORTH:SPACE':
            errorcat = 'SPACE'
        else:
            errorcat = 'ORTH'
    elif errtype == 'R:PRON':
        errorcat = errtype
    elif errtype == 'R:PART':
        errorcat = errtype
    elif errtype == 'R:PUNCT':
        if ori_tok == '.' and cor_tok == '?':
            errorcat = 'PERIODINSQUEST'
        elif ori_tok == ',' and cor_tok == '.':
            errorcat = 'COMSPLICE'
        else:
            errorcat = errtype
    elif errtype.startswith('R:ADJ'):
        if oper in ['$TRANSFORM_VERB_VBG_VBN','$TRANSFORM_VERB_VBD_VBN']:
            errorcat = 'PARTADJ'
        elif oper in ['$SUFFIXTRANSFORM_REMOVE_er', '$SUFFIXTRANSFORM_APPEND_er']:
            errorcat = 'R:ADJ:COMP'
        elif oper == ['$SUFFIXTRANSFORM_REMOVE_est','$SUFFIXTRANSFORM_APPEND_est']:
            errorcat = 'R:ADJ:SUPR'
        elif ori_tok.lower != cor_tok.lower() and ori_tok.lower() in ['many', 'much'] and cor_tok.lower() in ['many','much']:
            errorcat = 'R:MANYVSMUCH'
        elif ori_tok.lower != cor_tok.lower() and ori_tok.lower() in ['sometime', 'sometimes'] and cor_tok.lower() in ['sometime', 'sometimes']:
            errorcat = 'ADJ:sometimeVSsometimes'

        elif errtype in ['R:ADJ:COMP','R:ADJ:SUPR']:
            errorcat =errtype
        elif prev_ori_tok == 'more' and prev_cor_tok != 'more' and cor_tok.endswith('er'):
            errorcat = 'R:ADJ:COMP'
        elif prev_ori_tok == 'most' and prev_cor_tok != 'most' and cor_tok.endswith('est'):
            errorcat = 'R:ADJ:SUPR'
        elif ori_tok.lower != cor_tok.lower() and ori_tok.lower() in ['less','fewer'] and cor_tok.lower() in  ['less','fewer']:
            errorcat = 'R:LESSVSFEWER'
        elif (ori_tok.endswith('ed') and cor_tok.endswith('ing')) or (ori_tok.endswith('ing') and cor_tok.endswith('ed')):
            errorcat = 'ADJ:edVSing'
        elif (ori_tok.endswith('ic') and cor_tok.endswith('ical')) or (ori_tok.endswith('ical') and cor_tok.endswith('ic')):
            errorcat = 'ADJ:icVSical'
        else:
            errorcat = errtype
    elif errtype.startswith('R:ADV'):
        if ori_tok.lower != cor_tok.lower() and ori_tok.lower() in ['sometime', 'sometimes'] and cor_tok.lower() in ['sometime', 'sometimes']:
            errorcat = 'ADJ:sometimeVSsometimes'
        elif cor_pos == 'ADV' and oper == '$SUFFIXTRANSFORM_APPEND_ly' :
            errorcat ='R:ADV:APPNLY'
        else:
            errorcat=errtype
    elif errtype == 'R:WO':
        errorcat = errtype
    elif errtype.startswith("R:MORPH"):
        if 'VB' in ori_dpos[0] and 'VB' in cor_dpos[0] and len(ori_dpos) == 1 and len(cor_dpos) == 1 and ori_dpos!=cor_dpos:
            errorcat = f'VERB_{ori_dpos[0]}_{cor_dpos[0]}'
        elif cor_pos =="ADJ" and ori_pos =='ADV' and oper == '$SUFFIXTRANSFORM_REMOVE_ly':
            errorcat = "ADVVSADJ"
        elif cor_pos == 'ADV' and ori_pos == "ADJ" and oper == '$SUFFIXTRANSFORM_APPEND_ly':
            errorcat = "ADJVSADV"
        elif cor_pos == 'ADV' and oper == '$SUFFIXTRANSFORM_APPEND_ly' :
            errorcat ='R:ADV:APPNLY'

        elif oper in PLURAL_OPERATIONS:
            errorcat = 'SingularAfterPluralQuant'
        elif oper in SINGULAR_OPERATIONS:
            if cor_tok in uncountable_nouns:
                errorcat = 'UCPLURAL'
            else:
                errorcat = 'INCPLUR'
        elif errtype in ['R:MORPH:JJ_JJ','R:MORPH:VBG_JJ']:
            if (ori_tok.endswith('ic') and cor_tok.endswith('ical')) or (ori_tok.endswith('ical') and cor_tok.endswith('ic')):
                errorcat ='ADJ:icVSical'
            elif (ori_tok.endswith('ed') and cor_tok.endswith('ing')) or (ori_tok.endswith('ing') and cor_tok.endswith('ed')):
                errorcat ='ADJ:edVSing'
            else:
                errorcat= errtype
        elif (ori_pos=="ADJ" and  cor_pos =="NOUN") or (ori_pos =="NOUN" and  cor_pos =="ADJ"):
            errorcat = 'ADJVSNOUN'

        elif (ori_pos == 'VERB' or cor_pos == 'VERB') and 'VERB' in oper:
            errorcat = oper.replace('$TRANSFORM_', '')
        else:
            errorcat=errtype
    elif errtype == "R:CONJ":
        if (ori_tok.lower() in COORD_CONJ) or (cor_tok.lower() in COORD_CONJ):
            errorcat = "COORDCONJ"
        elif (ori_tok.lower() in SUBORD_CONJ) or (cor_tok.lower() in SUBORD_CONJ):
            errorcat = "SUBORDCONJ"
        elif ((ori_tok.lower() in CORRCONJ_1) or (cor_tok.lower() in CORRCONJ_1)) and ((ori_tok.lower() in CORRCONJ_2) or (cor_tok.lower() in CORRCONJ_2)):
            errorcat = "CORRCONJ"
        else:
            errorcat = errtype
    elif ori_tok in PREPOSITIONS and cor_tok in PREPOSITIONS:
        errorcat = 'R:PREP'
    else:
        errorcat = errtype
    return errorcat

def check_replacing_rules(errtype,ori_tok,cor_tok,ori_pos, cor_pos,start1, uncountable_nouns,verb_form_dict, prev_ori_tok, prev_cor_tok,ori_dpos, cor_dpos):
    oper=""
    if errtype == 'R:OTHER':
        if ori_tok in CONFUSED_WORDS.keys() and cor_tok in CONFUSED_WORDS.values():
            errorcat=f"{ori_tok.lower()}vs{cor_tok.lower()}"
        elif ori_tok in CONFUSED_WORDS.values() and cor_tok in CONFUSED_WORDS.keys():
            errorcat=f"{cor_tok.lower()}vs{ori_tok.lower()}"

        elif (ori_pos == 'ADJ' and cor_pos == 'ADV') or (ori_pos == 'ADV' and cor_pos == 'ADJ'):
            errorcat = 'ADJVSADV'
        elif 'VB' in ori_dpos and 'VB' in cor_dpos:
            errorcat = f'VERB_{ori_dpos}_{cor_dpos}'
        elif ori_tok[0] in [',', '.', '!', '?'] or cor_tok[0] in [',', '.', '!', '?'] and ori_tok[0] != cor_tok[0]:
            ori_tok1 = ori_tok[0]
            #ori_tok2 = ori_tok[1:].strip()
            cor_tok1 = cor_tok[0]
            #cor_tok2 = cor_tok[1:].strip()
            if ori_tok1 == ',' and cor_tok1 == '.':
                errorcat = 'COMSPLICE'
            elif ori_tok1 == '.' and cor_tok1 == '?':
                errorcat = 'PERIODINSQUEST'
            else:
                errorcat = 'R:PUNCT'
        else:
            errorcat = errtype

    elif errtype == 'R:PREP':
        errorcat = errtype
    elif 'R:NOUN' in errtype:
        if errtype == 'R:NOUN:POSS':
            errorcat = errtype
        elif errtype == 'R:NOUN:NUM':
            if cor_tok in uncountable_nouns:
                errorcat = 'UCPLURAL'
            else:
                errorcat = 'INCPLUR'
        else:
            errorcat = errtype
    elif errtype == 'R:SPELL':
        errorcat = errtype
    elif 'R:VERB' in errtype:
        if 'VB' in ori_dpos[0] and 'VB' in cor_dpos[0] and len(ori_dpos)==1 and len(cor_dpos) ==1 and ori_dpos!=cor_dpos:
            errorcat = f'VERB_{ori_dpos[0]}_{cor_dpos[0]}'
        elif errtype == 'R:VERB:SVA':
            errorcat = errtype
        elif (cor_tok in DO and ori_tok in MAKE) or (
                ori_tok in DO and cor_tok in MAKE):
            errorcat = 'COMFUSEDOMAKE'
        elif errtype == 'R:VERB:INFL':
            errorcat = 'R:INFL'
        else:
            oper = get_operation.apply_transformation(ori_tok, cor_tok,verb_form_dict)
            if oper and '$TRANSFORM_VERB' in oper:
                errorcat = oper.replace('$TRANSFORM_', '')
            else:
                errorcat = errtype
    elif 'R:DET' in errtype:
        errorcat = 'IncoArt'
    elif errtype.startswith('R:ORTH'):
        errorcat = errtype
        if errtype == 'R:ORTH:Capt1st':
            if start1 == 0:
                errorcat = 'R:ORTH:Capt1stWord'
            elif ori_tok == 'i':
                errorcat = 'LowPronI'
        elif errtype == 'R:ORTH:HYPHEN':
            errorcat = 'HYPHEN'
        elif errtype == 'R:ORTH:SPACE':
            errorcat = 'SPACE'
        else:
            errorcat = errtype
    elif errtype == 'R:PRON':
        errorcat = errtype
    elif errtype == 'R:PART':
        errorcat = errtype
    elif errtype == 'R:PUNCT':
        if ori_tok == '.' and cor_tok == '?':
            errorcat = 'PERIODINSQUEST'
        elif ori_tok == ',' and cor_tok == '.':
            errorcat = 'COMSPLICE'
        else:
            errorcat = errtype
    elif errtype.startswith('R:ADJ'):
        if ori_tok.lower != cor_tok.lower() and ori_tok.lower() in ['many', 'much'] and cor_tok.lower() in ['many','much']:
            errorcat = 'R:MANYVSMUCH'
        elif errtype in ['R:ADJ:COMP','R:ADJ:SUPR']:
            errorcat =errtype
        elif prev_ori_tok == 'more' and prev_cor_tok != 'more' and cor_tok.endswith('er'):
            errorcat = 'R:ADJ:COMP'
        elif prev_ori_tok == 'most' and prev_cor_tok != 'most' and cor_tok.endswith('est'):
            errorcat = 'R:ADJ:SUPR'
        elif ori_tok.lower != cor_tok.lower() and ori_tok.lower() in ['less','fewer'] and cor_tok.lower() in  ['less','fewer']:
            errorcat = 'R:LESSVSFEWER'
        elif (ori_tok.endswith('ed') and cor_tok.endswith('ing')) or (ori_tok.endswith('ing') and cor_tok.endswith('ed')):
            errorcat = 'ADJ:edVSing'
        elif (ori_tok.endswith('ic') and cor_tok.endswith('ical')) or (ori_tok.endswith('ical') and cor_tok.endswith('ic')):
            errorcat = 'ADJ:icVSical'
        else:
            errorcat = errtype
    elif errtype.startswith('R:ADV'):
        if ori_tok.lower != cor_tok.lower() and ori_tok.lower() in ['sometime', 'sometimes'] and cor_tok.lower() in ['sometime', 'sometimes']:
            errorcat = 'ADJ:sometimeVSsometimes'
        else:
            errorcat=errtype
    elif errtype == 'R:WO':
        errorcat = errtype
    elif errtype.startswith("R:MORPH"):
        if 'VB' in ori_dpos[0] and 'VB' in cor_dpos[0] and len(ori_dpos) == 1 and len(
                cor_dpos) == 1 and ori_dpos != cor_dpos:
            errorcat = f'VERB_{ori_dpos[0]}_{cor_dpos[0]}'
        elif cor_pos == "ADJ" and ori_pos == 'ADV':
            errorcat = "ADVVSADJ"
        elif cor_pos == 'ADV' and ori_pos == "ADJ":
            errorcat = "ADJVSADV"
        elif errtype in ['R:MORPH:JJ_JJ', 'R:MORPH:VBG_JJ']:
            if (ori_tok.endswith('ic') and cor_tok.endswith('ical')) or (
                    ori_tok.endswith('ical') and cor_tok.endswith('ic')):
                errorcat = 'ADJ:icVSical'
            elif (ori_tok.endswith('ed') and cor_tok.endswith('ing')) or (
                    ori_tok.endswith('ing') and cor_tok.endswith('ed')):
                errorcat = 'ADJ:edVSing'
            else:
                errorcat = errtype
        elif ori_pos == "ADJ" and cor_pos == "NOUN" or ori_pos == "NOUN" and cor_pos == "ADJ":
            errorcat = 'ADJVSNOUN'

        elif (ori_pos == 'VERB' or cor_pos == 'VERB') and 'VERB' in oper:
            errorcat = oper.replace('$TRANSFORM_', '')
        else:
            errorcat = errtype
    elif errtype == "R:CONJ":
        if (ori_tok.lower() in COORD_CONJ) or (cor_tok.lower() in COORD_CONJ):
            errorcat = "COORDCONJ"
        elif (ori_tok.lower() in SUBORD_CONJ) or (cor_tok.lower() in SUBORD_CONJ):
            errorcat = "SUBORDCONJ"
        elif ((ori_tok.lower() in CORRCONJ_1) or (cor_tok.lower() in CORRCONJ_1)) and ((ori_tok.lower() in CORRCONJ_2) or (cor_tok.lower() in CORRCONJ_2)):
            errorcat = "CORRCONJ"
        else:
            errorcat = errtype
    elif ori_tok in PREPOSITIONS and cor_tok in PREPOSITIONS:
        errorcat = 'R:PREP'
    else:
        errorcat = errtype
    return errorcat,oper

def check_unneccessary(errtype,ori_tok):
    if errtype == 'U:PUNCT':
        if ori_tok == ',':
            errorcat = 'U:COMMA'
        elif ori_tok == '?':
            errorcat = 'U:QUEST'
        else:
            errorcat = errtype
    elif errtype=='U:DET':
        if ori_tok.lower() in ARTICLES:
            errorcat = 'U:ART'
        else:
            errorcat = 'DETUSE'
    elif errtype.startswith("U:VERB"):
        errorcat = 'U:VERB'
    elif errtype.startswith("U:NOUN"):
        errorcat = 'U:NOUN'
    elif ori_tok in PREPOSITIONS:
        errorcat = errtype
    else:
        errorcat = "U:WORD"
    return errorcat

def combine_edits(gector_edit,rule_edits,uncountable_nouns,lang,verb_form_dict,context_based_dict, detailed_explanatory=False,general_error_exp_dict=None):
    edits=[]
    mismatch=False
    if len(rule_edits) == len(gector_edit):
        for edit1, edit2 in zip(rule_edits,gector_edit):
            start1,end1, errtype, ori_tok, cor_tok, ori_pos, cor_pos, start2, end2, prev_ori_tok, prev_cor_tok, ori_dpos, cor_dpos = edit1
            #startg1,endg1,cor_tokg,_,oper,ori_tokg = edit2
            startg1,endg1,cor_tokg,oper,ori_tokg = edit2

            if start1 == startg1 and end1 == endg1:
                if errtype.startswith('M:'):
                    errorcat = check_missing_gect(errtype,ori_tok,cor_tok,ori_tokg, oper)
                elif errtype.startswith('R:'):
                    errorcat = check_replacing_gect(errtype,ori_tok,cor_tok,ori_tokg,ori_pos[0],cor_pos[0], oper,start1,uncountable_nouns, prev_ori_tok, prev_cor_tok, ori_dpos, cor_dpos)
                elif errtype.startswith('U:'):
                    errorcat = check_unneccessary(errtype, ori_tok)
                else:
                    errorcat = 'Others'
                edits.append(format_errordetails_dictedits(start1, end1, errtype, oper, errorcat, ori_tok, cor_tok, ori_tokg,cor_tokg, start2, end2,detailed_explanatory,context_based_dict, general_error_exp_dict,lang))
            else:
                mismatch=True
                if errtype.startswith('M:'):
                    errorcat = check_missing_rules(errtype, ori_tok,cor_tok)
                elif errtype.startswith('R:'):
                    errorcat = check_replacing_rules(errtype, ori_tok, cor_tok,ori_pos[0], cor_pos[0], start1, uncountable_nouns,verb_form_dict, prev_ori_tok, prev_cor_tok,ori_dpos, cor_dpos)
                elif errtype.startswith('U:'):
                    errorcat = check_unneccessary(errtype, ori_tok)
                else:
                    errorcat = 'Others'
                edits.append(format_errordetails_dictedits(start1, end1, errtype, oper, errorcat, ori_tok, cor_tok, ori_tokg,cor_tokg, start2, end2, detailed_explanatory, context_based_dict, general_error_exp_dict,lang))
    else:
        mismatch=True
        for edit1 in rule_edits:
            oper=""
            start1, end1, errtype, ori_tok, cor_tok, ori_pos, cor_pos, start2, end2, prev_ori_tok, prev_cor_tok, ori_dpos, cor_dpos  = edit1
            if errtype.startswith('M:'):
                errorcat = check_missing_rules(errtype,ori_tok, cor_tok)
            elif errtype.startswith('R:'):
                errorcat,oper = check_replacing_rules(errtype, ori_tok, cor_tok,ori_pos[0],cor_pos[0],start1,uncountable_nouns, verb_form_dict,prev_ori_tok, prev_cor_tok,ori_dpos, cor_dpos)
            elif errtype.startswith('U:'):
                errorcat = check_unneccessary(errtype, ori_tok)
            else:
                errorcat = 'Others'
            edits.append(format_errordetails_dictedits(start1, end1, errtype, oper, errorcat, ori_tok, cor_tok, '','', start2, end2,detailed_explanatory,context_based_dict,general_error_exp_dict,lang))
    return edits, mismatch

def par_to_edit(orig_sent,cor_sent,gb_spell,tag_map, args,nlp):
    #print(os.system('which python'))
    # Get base working directory.
    edits=[]
    #basename = os.path.dirname(os.path.realpath(__file__))
    # Load Tokenizer and other resources
    # Lancaster Stemmer
    stemmer = LancasterStemmer()
    # GB English word list (inc -ise and -ize)
    #gb_spell = toolbox.loadDictionary(basename+"/resources/en_GB-large.txt")
    # Part of speech map file
    #tag_map = toolbox.loadTagMap(basename+"/resources/en-ptb_map")
    orig_sent = orig_sent.strip()
    #orig_sent_tok = nlp.tokenizer(orig_sent)
    proc_orig = toolbox.applySpacy(orig_sent.split(), nlp)
    # Loop through the corrected sentences
    cor_sent = cor_sent.strip()
    #cor_sent_tok = nlp.tokenizer(cor_sent)

    # Identical sentences have no edits, so just write noop.
    if orig_sent == cor_sent:
        edits.append([0, 0, '$CORRECT', '', 0, 0])
    # Otherwise, do extra processing.
    else:
        # Markup the corrected sentence with spacy (assume tokenized)
        proc_cor = toolbox.applySpacy(cor_sent.split(), nlp)
        # Auto align the parallel sentences and extract the edits.
        auto_edits = align_text.getAutoAlignedEdits(proc_orig, proc_cor, args)
        # Loop through the edits.
        for auto_edit in auto_edits:
            # Give each edit an automatic error type.
            if auto_edit[2] =="CORRECT":
                edits.append(auto_edit)
                continue
            cat,orig_pos,cor_pos, ori_dpos, cor_dpos, prev_ori_tok, prev_cor_tok,_,_,_,_,_,_ = cat_rules.autoTypeEdit(auto_edit, proc_orig, proc_cor, gb_spell, tag_map, nlp, stemmer)
            if cat.startswith('M'):
                auto_edit[0]=auto_edit[0]-1
                auto_edit[1]=auto_edit[0]+1
            auto_edit[2] = cat
            auto_edit[5] = orig_pos
            auto_edit[6] = cor_pos
            auto_edit[9] = prev_ori_tok
            auto_edit[10] = prev_cor_tok
            auto_edit[11] = ori_dpos
            auto_edit[12] = cor_dpos
            # Write the edit to the output m2 file.
            edits.append(auto_edit)
    return edits

