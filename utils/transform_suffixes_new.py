"""Transform functions for replace."""
import pickle

def is_append_suffix(suffix,w1,w2):
  l = len(suffix)
  if (w2[-l:] == suffix) and (w1 == w2[:-l]):
    return True
  else:
    return False

def is_transform_suffix(suffix_1,suffix_2, w1, w2):
    l1 = len(suffix_1)
    l2 = len(suffix_2)

    if (w1[-l1:] == suffix_1) and (w2[-l2:] == suffix_2) and (w1[:-l1] == w2[:-l2]):
        return True
    else:
        return False

def append_suffix(word, suffix):

    #print("word: {}, suffix: {}".format(word, suffix))

    if len(word) == 1:
        print("We currently fear appending suffix: {} to a one letter word, but still doing it: {}".format(suffix, word))
        #return word

    l = len(suffix)

    #if word[-l:] == suffix:
        #print("**** WARNING: SUFFIX: {} ALREADY PRESENT in WORD, BUT still adding it: {} ****".format(suffix, word))

    
    if word[-1] == "s" and suffix == "s":
        return word+"es"

    if word[-1] == "y" and suffix == "s" and len(word)>2:
        if word[-2] not in ["a","e","i","o","u"]:
            return word[0:-1] + "ies"

    
    #if word[-1] == "h" and suffix == "s":
    #    return word + "es"

    #if word[-1] == "t" and suffix == "d":
    #    return word + "ed"

    #if word[-1] == "k" and suffix == "d":
    #    return word + "ed"    

    return word+suffix

def remove_suffix(word, suffix):

    if len(suffix) > len(word):
        print("suffix: {} to be removed has larger length than word: {}".format(suffix, word))
        return word

    l = len(suffix)

    if word[-l:] == suffix:
        return word[:-l]
    else:
        print("**** WARNING: SUFFIX : {} NOT PRESENT in WORD: {} ****".format(suffix, word))
        return word

def transform_suffix(word, suffix_1, suffix_2):
    
    if len(suffix_1) > len(word):
        print("suffix: {} to be replaced has larger length than word: {}".format(suffix_1, word))
        return word

    l1 = len(suffix_1)
    l2 = len(suffix_2)

    if word[-l1:] == suffix_1:
        return word[:-l1] + suffix_2
    else:
        print("transform")
        print("**** WARNING: SUFFIX : {} NOT PRESENT in WORD: {} ****".format(suffix_1, word))
        return word


class SuffixTransform():
    """Helper to find if a replacement in a sentence matches a predefined transform."""

    def __init__(self, src_word, tgt_word):
        """Create a new transform matching instance
           :src_word : original word
           :tgt_word : modified word
        """
        self.src_word = src_word
        self.tgt_word = tgt_word

    def match(self):
        """Returns an opcode if matches."""
        if self.src_word == self.tgt_word:
            return None

        '''
        return self.pluralization() or self.singularization() \
             or self.capitalization() or self.decapitalization() \
             or self.verb_transform() or None
        '''

        return self.append_s() or self.remove_s() \
        	or self.append_d() or self.remove_d() \
        	or self.append_es() or self.remove_es() \
        	or self.append_ing() or self.remove_ing() \
        	or self.append_ed() or self.remove_ed() \
        	or self.append_ly() or self.remove_ly() \
        	or self.append_er() or self.remove_er() \
        	or self.append_al() or self.remove_al() \
        	or self.append_n() or self.remove_n() \
        	or self.append_y() or self.remove_y() \
            or self.append_ation() or self.remove_ation() \
        	or self.e_to_ing() or self.ing_to_e() \
        	or self.d_to_t() or self.t_to_d() \
        	or self.d_to_s() or self.s_to_d() \
        	or self.s_to_ing() or self.ing_to_s() \
        	or self.n_to_ing() or self.ing_to_n() \
        	or self.nce_to_t() or self.t_to_nce() \
            or self.s_to_ed() or self.ed_to_s() \
            or self.ing_to_ed() or self.ed_to_ing() \
            or self.ing_to_ion() or self.ion_to_ing() \
            or self.ing_to_ation() or self.ation_to_ing() \
            or self.t_to_ce() or self.ce_to_t() \
            or self.y_to_ic() or self.ic_to_y() \
            or self.t_to_s() or self.s_to_t() \
            or self.e_to_al() or self.al_to_e() \
            or self.y_to_ily() or self.ily_to_y() \
            or self.y_to_ied() or self.ied_to_y() \
            or self.y_to_ical() or self.ical_to_y() \
            or self.y_to_ies() or self.ies_to_y() \
            or self.er_to_est() or self.est_to_er()  \
            or self.append_est() or self.remove_est() \
            or self.append_ism() or self.remove_ism()  \
            or self.append_ship() or self.append_age()  \
            or self.remove_age() or self.append_ness()  \
            or self.remove_ness() or self.append_ive()  \
            or self.remove_ive() or self.append_ful()  \
            or self.remove_ful() or self.append_less()  \
            or self.remove_less() or self.append_able()  \
            or self.remove_able() or self.append_ist()  \
            or self.remove_ist() or self.append_wise()  \
            or self.remove_wise() \
            or None



    def e_to_ing(self):
        if is_transform_suffix("e","ing",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_E_TO_ING"
        else:
            return None

    def ing_to_e(self):
        if is_transform_suffix("ing","e",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ING_TO_E"
        else:
            return None

    def d_to_t(self):
        if is_transform_suffix("d","t",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_D_TO_T"
        else:
            return None

    def t_to_d(self):
        if is_transform_suffix("t","d",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_T_TO_D"
        else:
            return None

    def d_to_s(self):
        if is_transform_suffix("d","s",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_D_TO_S"
        else:
            return None

    def s_to_d(self):
        if is_transform_suffix("s","d",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_S_TO_D"
        else:
            return None

    def s_to_ing(self):
        if is_transform_suffix("s","ing",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_S_TO_ING"
        else:
            return None

    def ing_to_s(self):
        if is_transform_suffix("ing","s",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ING_TO_S"
        else:
            return None

    def n_to_ing(self):
        if is_transform_suffix("n","ing",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_N_TO_ING"
        else:
            return None

    def ing_to_n(self):
        if is_transform_suffix("ing","n",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ING_TO_N"
        else:
            return None

    def t_to_nce(self):
        if is_transform_suffix("t","nce",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_T_TO_NCE"
        else:
            return None

    def nce_to_t(self):
        if is_transform_suffix("nce","t",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_NCE_TO_T"
        else:
            return None

    def s_to_ed(self):
        if is_transform_suffix("s","ed",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_S_TO_ED"
        else:
            return None

    def ed_to_s(self):
        if is_transform_suffix("ed","s",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ED_TO_S"
        else:
            return None

    def ing_to_ed(self):
        if is_transform_suffix("ing","ed",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ING_TO_ED"
        else:
            return None

    def ed_to_ing(self):
        if is_transform_suffix("ed","ing",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ED_TO_ING"
        else:
            return None

    def er_to_est(self):
        if is_transform_suffix("er","est",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ER_TO_EST"
        else:
            return None

    def est_to_er(self):
        if is_transform_suffix("est","er",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_EST_TO_ER"
        else:
            return None

    def ing_to_ion(self):
        if is_transform_suffix("ing","ion",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ING_TO_ION"
        else:
            return None

    def ion_to_ing(self):
        if is_transform_suffix("ion","ing",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ION_TO_ING"
        else:
            return None

    def ing_to_ation(self):
        if is_transform_suffix("ing","ation",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ING_TO_ATION"
        else:
            return None

    def ation_to_ing(self):
        if is_transform_suffix("ation","ing",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ATION_TO_ING"
        else:
            return None

    def t_to_ce(self):
        if is_transform_suffix("t","ce",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_T_TO_CE"
        else:
            return None

    def ce_to_t(self):
        if is_transform_suffix("ce","t",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_CE_TO_T"
        else:
            return None

    def y_to_ic(self):
        if is_transform_suffix("y","ic",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_Y_TO_IC"
        else:
            return None

    def ic_to_y(self):
        if is_transform_suffix("ic","y",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_IC_TO_Y"
        else:
            return None

    def t_to_s(self):
        if is_transform_suffix("t","s",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_T_TO_S"
        else:
            return None

    def s_to_t(self):
        if is_transform_suffix("s","t",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_S_TO_T"
        else:
            return None

    def e_to_al(self):
        if is_transform_suffix("e","al",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_E_TO_AL"
        else:
            return None

    def al_to_e(self):
        if is_transform_suffix("al","e",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_AL_TO_E"
        else:
            return None

    def y_to_ily(self):
        if is_transform_suffix("y","ily",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_Y_TO_ILY"
        else:
            return None

    def ily_to_y(self):
        if is_transform_suffix("ily","y",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ILY_TO_Y"
        else:
            return None

    def y_to_ied(self):
        if is_transform_suffix("y","ied",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_Y_TO_IED"
        else:
            return None

    def ied_to_y(self):
        if is_transform_suffix("ied","y",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_IED_TO_Y"
        else:
            return None

    def y_to_ical(self):
        if is_transform_suffix("y","ical",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_Y_TO_ICAL"
        else:
            return None

    def ical_to_y(self):
        if is_transform_suffix("ical","y",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_ICAL_TO_Y"
        else:
            return None

    def y_to_ies(self):
        if is_transform_suffix("y","ies",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_Y_TO_IES"
        else:
            return None

    def ies_to_y(self):
        if is_transform_suffix("ies","y",self.src_word,self.tgt_word):
            return "$SUFFIXTRANSFORM_IES_TO_Y"
        else:
            return None

    def append_s(self):
        if is_append_suffix("s",self.src_word, self.tgt_word) and (self.src_word not in ["a","A","I","i"]):
            return "$SUFFIXTRANSFORM_APPEND_s"
        else:
            return None

    def remove_s(self):
        if is_append_suffix("s",self.tgt_word,self.src_word) and (self.src_word not in ["As","as","is","Is"]):
            return "$SUFFIXTRANSFORM_REMOVE_s"
        else:
            return None

    def append_d(self):
        if is_append_suffix("d",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_d"
        else:
            return None

    def remove_d(self):
        if is_append_suffix("d",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_d"
        else:
            return None

    def append_es(self):
        if is_append_suffix("es",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_es"
        else:
            return None

    def append_est(self):
        if is_append_suffix("est",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_est"
        else:
            return None
    def remove_es(self):
        if is_append_suffix("es",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_es"
        else:
            return None

    def remove_est(self):
        if is_append_suffix("est",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_est"
        else:
            return None
    def append_ing(self):
        if is_append_suffix("ing",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ing"
        else:
            return None

    def remove_ing(self):
        if is_append_suffix("ing",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ing"
        else:
            return None

    def append_ed(self):
        if is_append_suffix("ed",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ed"
        else:
            return None

    def remove_ed(self):
        if is_append_suffix("ed",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ed"
        else:
            return None


    def append_ism(self):
        if is_append_suffix("ism",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ism"
        else:
            return None

    def remove_ism(self):
        if is_append_suffix("ism",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ism"
        else:
            return None


    def append_ship(self):
        if is_append_suffix("ship",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ship"
        else:
            return None

    def append_age(self):
        if is_append_suffix("age",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_age"
        else:
            return None

    def remove_age(self):
        if is_append_suffix("age",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_age"
        else:
            return None

    def append_ness(self):
        if is_append_suffix("ness",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ness"
        else:
            return None

    def remove_ness(self):
        if is_append_suffix("ness",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ness"
        else:
            return None


    def append_ive(self):
        if is_append_suffix("ive",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ive"
        else:
            return None

    def remove_ive(self):
        if is_append_suffix("ive",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ive"
        else:
            return None


    def append_ful(self):
        if is_append_suffix("ful",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ful"
        else:
            return None

    def remove_ful(self):
        if is_append_suffix("ful",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ful"
        else:
            return None

    def append_less(self):
        if is_append_suffix("less",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_less"
        else:
            return None

    def remove_less(self):
        if is_append_suffix("less",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_less"
        else:
            return None

    def append_able(self):
        if is_append_suffix("able",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_able"
        else:
            return None

    def remove_able(self):
        if is_append_suffix("able",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_able"
        else:
            return None

    def append_ist(self):
        if is_append_suffix("ist",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ist"
        else:
            return None

    def remove_ist(self):
        if is_append_suffix("ist",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ist"
        else:
            return None

    def append_wise(self):
        if is_append_suffix("wise",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_wise"
        else:
            return None

    def remove_wise(self):
        if is_append_suffix("wise",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_wise"
        else:
            return None

    def remove_ed(self):
        if is_append_suffix("ed",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ed"
        else:
            return None


    def append_ly(self):
        if is_append_suffix("ly",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ly"
        else:
            return None

    def remove_ly(self):
        if is_append_suffix("ly",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ly"
        else:
            return None

    def append_er(self):
        if is_append_suffix("er",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_er"
        else:
            return None

    def remove_er(self):
        if is_append_suffix("er",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_er"
        else:
            return None

    def append_al(self):
        if is_append_suffix("al",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_al"
        else:
            return None

    def remove_al(self):
        if is_append_suffix("al",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_al"
        else:
            return None

    def append_n(self):
        if is_append_suffix("n",self.src_word, self.tgt_word) and (self.src_word not in ["a","A","i","I"]):
            return "$SUFFIXTRANSFORM_APPEND_n"
        else:
            return None

    def remove_n(self):
        if is_append_suffix("n",self.tgt_word,self.src_word) and (self.src_word not in ["an","An","in","In"]):
            return "$SUFFIXTRANSFORM_REMOVE_n"
        else:
            return None

    def append_y(self):
        if is_append_suffix("y",self.src_word, self.tgt_word) and (self.src_word not in ["m","M"]):
            return "$SUFFIXTRANSFORM_APPEND_y"
        else:
            return None
    def remove_y(self):
        if is_append_suffix("y",self.tgt_word,self.src_word) and (self.src_word not in ["My","my"]):
            return "$SUFFIXTRANSFORM_REMOVE_y"
        else:
            return None

    def append_ation(self):
        if is_append_suffix("ation",self.src_word, self.tgt_word):
            return "$SUFFIXTRANSFORM_APPEND_ation"
        else:
            return None

    def remove_ation(self):
        if is_append_suffix("ation",self.tgt_word,self.src_word):
            return "$SUFFIXTRANSFORM_REMOVE_ation"
        else:
            return None

def apply_suffixtransform(uncorrected, opcode):
    """Tries to apply an opcode to a word or returns None
        :param uncorrected: Tokenized uncorrected sentence
        :uposition: Position of replaced in uncorrected sentence
        :opcode: Opcode to try applying
    """
    art = ApplySuffixTransorm(uncorrected, opcode)
    return art.apply()

class ApplySuffixTransorm():

    def __init__(self, uncorrected, opcode):
        """Tries to apply an opcode to a word or returns None
            :param uncorrected: Tokenized uncorrected sentence
            :uposition: Position of replaced in uncorrected sentence
            :opcode: Opcode to try applying
        """

        self.uncorrected = uncorrected
        self.opcode = opcode
        self.src_word = uncorrected

    def apply(self):
        """Try to apply the transform
            :return: Transformed word or None if cannot transform
        """
        transformed = None

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_s":
            transformed = append_suffix(self.src_word, "s")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_d":
            transformed = append_suffix(self.src_word, "d")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_es":
            transformed = append_suffix(self.src_word, "es")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ing":
            transformed = append_suffix(self.src_word, "ing")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ed":
            transformed = append_suffix(self.src_word, "ed")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ly":
            transformed = append_suffix(self.src_word, "ly")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_er":
            transformed = append_suffix(self.src_word, "er")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_al":
            transformed = append_suffix(self.src_word, "al")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_n":
            transformed = append_suffix(self.src_word, "n")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_y":
            transformed = append_suffix(self.src_word, "y")

        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ation":
            transformed = append_suffix(self.src_word, "ation")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_s":
            transformed = remove_suffix(self.src_word, "s")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_d":
            transformed = remove_suffix(self.src_word, "d")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_es":
            transformed = remove_suffix(self.src_word, "es")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ing":
            transformed = remove_suffix(self.src_word, "ing")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ed":
            transformed = remove_suffix(self.src_word, "ed")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ly":
            transformed = remove_suffix(self.src_word, "ly")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_er":
            transformed = remove_suffix(self.src_word, "er")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_al":
            transformed = remove_suffix(self.src_word, "al")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_n":
            transformed = remove_suffix(self.src_word, "n")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_y":
            transformed = remove_suffix(self.src_word, "y")

        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ation":
            transformed = remove_suffix(self.src_word, "ation")

        if self.opcode == "$SUFFIXTRANSFORM_E_TO_ING":
            transformed = transform_suffix(self.src_word, "e", "ing")

        if self.opcode == "$SUFFIXTRANSFORM_ING_TO_E":
            transformed = transform_suffix(self.src_word, "ing", "e")

        if self.opcode == "$SUFFIXTRANSFORM_D_TO_T":
            transformed = transform_suffix(self.src_word, "d", "t")

        if self.opcode == "$SUFFIXTRANSFORM_T_TO_D":
            transformed = transform_suffix(self.src_word, "t", "d")

        if self.opcode == "$SUFFIXTRANSFORM_D_TO_S":
            transformed = transform_suffix(self.src_word, "d", "s")

        if self.opcode == "$SUFFIXTRANSFORM_S_TO_D":
            transformed = transform_suffix(self.src_word, "s", "d")

        if self.opcode == "$SUFFIXTRANSFORM_S_TO_ING":
            transformed = transform_suffix(self.src_word, "s", "ing")

        if self.opcode == "$SUFFIXTRANSFORM_ING_TO_S":
            transformed = transform_suffix(self.src_word, "ing", "s")

        if self.opcode == "$SUFFIXTRANSFORM_N_TO_ING":
            transformed = transform_suffix(self.src_word, "n", "ing")

        if self.opcode == "$SUFFIXTRANSFORM_ING_TO_N":
            transformed = transform_suffix(self.src_word, "ing", "n")

        if self.opcode == "$SUFFIXTRANSFORM_T_TO_NCE":
            transformed = transform_suffix(self.src_word, "t", "nce")

        if self.opcode == "$SUFFIXTRANSFORM_NCE_TO_T":
            transformed = transform_suffix(self.src_word, "nce", "t")

        if self.opcode == "$SUFFIXTRANSFORM_S_TO_ED":
            transformed = transform_suffix(self.src_word, "s", "ed")

        if self.opcode == "$SUFFIXTRANSFORM_ED_TO_S":
            transformed = transform_suffix(self.src_word, "ed", "s")

        if self.opcode == "$SUFFIXTRANSFORM_ING_TO_ED":
            transformed = transform_suffix(self.src_word, "ing", "ed")

        if self.opcode == "$SUFFIXTRANSFORM_ED_TO_ING":
            transformed = transform_suffix(self.src_word, "ed", "ing")

        if self.opcode == "$SUFFIXTRANSFORM_ING_TO_ION":
            transformed = transform_suffix(self.src_word, "ing", "ion")

        if self.opcode == "$SUFFIXTRANSFORM_ION_TO_ING":
            transformed = transform_suffix(self.src_word, "ion", "ing")

        if self.opcode == "$SUFFIXTRANSFORM_ING_TO_ATION":
            transformed = transform_suffix(self.src_word, "ing", "ation")

        if self.opcode == "$SUFFIXTRANSFORM_ATION_TO_ING":
            transformed = transform_suffix(self.src_word, "ation", "ing")

        if self.opcode == "$SUFFIXTRANSFORM_T_TO_CE":
            transformed = transform_suffix(self.src_word, "t", "ce")

        if self.opcode == "$SUFFIXTRANSFORM_CE_TO_T":
            transformed = transform_suffix(self.src_word, "ce", "t")

        if self.opcode == "$SUFFIXTRANSFORM_Y_TO_IC":
            transformed = transform_suffix(self.src_word, "y", "ic")

        if self.opcode == "$SUFFIXTRANSFORM_IC_TO_Y":
            transformed = transform_suffix(self.src_word, "ic", "y")

        if self.opcode == "$SUFFIXTRANSFORM_T_TO_S":
            transformed = transform_suffix(self.src_word, "t", "s")

        if self.opcode == "$SUFFIXTRANSFORM_S_TO_T":
            transformed = transform_suffix(self.src_word, "s", "t")

        if self.opcode == "$SUFFIXTRANSFORM_E_TO_AL":
            transformed = transform_suffix(self.src_word, "e", "al")

        if self.opcode == "$SUFFIXTRANSFORM_AL_TO_E":
            transformed = transform_suffix(self.src_word, "al", "e")

        if self.opcode == "$SUFFIXTRANSFORM_Y_TO_ILY":
            transformed = transform_suffix(self.src_word, "y", "ily")

        if self.opcode == "$SUFFIXTRANSFORM_ILY_TO_Y":
            transformed = transform_suffix(self.src_word, "ily", "y")

        if self.opcode == "$SUFFIXTRANSFORM_Y_TO_IED":
            transformed = transform_suffix(self.src_word, "y", "ied")

        if self.opcode == "$SUFFIXTRANSFORM_IED_TO_Y":
            transformed = transform_suffix(self.src_word, "ied", "y")

        if self.opcode == "$SUFFIXTRANSFORM_Y_TO_ICAL":
            transformed = transform_suffix(self.src_word, "y", "ical")

        if self.opcode == "$SUFFIXTRANSFORM_ICAL_TO_Y":
            transformed = transform_suffix(self.src_word, "ical", "y")

        if self.opcode == "$SUFFIXTRANSFORM_Y_TO_IES":
            transformed = transform_suffix(self.src_word, "y", "ies")

        if self.opcode == "$SUFFIXTRANSFORM_IES_TO_Y":
            transformed = transform_suffix(self.src_word, "ies", "y")

        if self.opcode == "$SUFFIXTRANSFORM_ER_TO_EST":
            transformed = transform_suffix(self.src_word, "er", "est")
        if self.opcode == "$SUFFIXTRANSFORM_EST_TO_ER":
            transformed = transform_suffix(self.src_word, "est", "er")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_est":
            transformed = remove_suffix(self.src_word, "est")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ism":
            transformed = remove_suffix(self.src_word, "ism")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ship":
            transformed = remove_suffix(self.src_word, "ship")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_age":
            transformed = remove_suffix(self.src_word, "age")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ness":
            transformed = remove_suffix(self.src_word, "ness")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ive":
            transformed = remove_suffix(self.src_word, "ive")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ful":
            transformed = remove_suffix(self.src_word, "ful")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_less":
            transformed = remove_suffix(self.src_word, "less")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_able":
            transformed = remove_suffix(self.src_word, "able")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_ist":
            transformed = remove_suffix(self.src_word, "ist")
        if self.opcode == "$SUFFIXTRANSFORM_REMOVE_wise":
            transformed = remove_suffix(self.src_word, "wise")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_est":
            transformed = append_suffix(self.src_word, "est")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ism":
            transformed = append_suffix(self.src_word, "ism")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ship":
            transformed = append_suffix(self.src_word, "ship")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_age":
            transformed = append_suffix(self.src_word, "age")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ness":
            transformed = append_suffix(self.src_word, "ness")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ive":
            transformed = append_suffix(self.src_word, "ive")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ful":
            transformed = append_suffix(self.src_word, "ful")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_less":
            transformed = append_suffix(self.src_word, "less")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_able":
            transformed = append_suffix(self.src_word, "able")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_ist":
            transformed = append_suffix(self.src_word, "ist")
        if self.opcode == "$SUFFIXTRANSFORM_APPEND_wise":
            transformed = append_suffix(self.src_word, "wise")
        return transformed or None