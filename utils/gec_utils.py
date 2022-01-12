# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import re
import nltk

paragraph_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x.split(",")))
    except TypeError:
        return [type(x)]


def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default

def replace_different_space(text):
    text = re.sub("\s", " ", text)
    return text


def is_sent(text):
    """
    :param text:
    :return:
    """
    if text.endswith(" ."):
        return True
    if text.endswith(" ?"):
        return True
    if text.endswith(" !"):
        return True
    return False


def space_puncts(sent):
    """e.g., noise!He -> noise ! He  影响分句的标点处理
    """
    _sent = []
    for word in sent.split(" "):
        detect = re.search("[A-Za-z][!\?\.]|[!\?\.][A-Za-z]", word)
        if detect is not None:
            split_tag = True
            name_suffix = ["Mr.", "Mrs.", "Ms.", "Miss.", "A.", "B.", "C.", "D.", "Dr.",
                           "adj.", "prep.", "adv.", "conj.", "P.S"]
            # n. v. "sb.", "sth.", "vt." 可能会造成原有该分句的，没被正确分句
            for suffix in name_suffix:
                if suffix in word:
                    split_tag = False
                    break
            common = set(word) & set("0123456789")
            if not word.count(".") > 1 and len(common) == 0 and split_tag:  # e.g., `U.S.` is correct.
                word = re.sub("([!\?\.])", r" \1 ", word)
        _sent.append(word)
    return " ".join(_sent)


def space_puncts_1(sent):
    """其他不影响分句的标点处理
    """
    _sent = []
    for word in sent.split(" "):
        detect = re.search("[A-Za-z][\?!,%\"#$&*=;@\\\|/]|[\?!,%\"#$&*=;@\\\|/][A-Za-z]", word)
        if detect is not None:
            common = set(word) & set("0123456789")
            if len(common) == 0:
                word = re.sub("([\?!,%\"#$&*=;@\\\|/])", r" \1 ", word)
        _sent.append(word)
    return " ".join(_sent)


def before_split_text(text):
    """
     for PPT
    name_suffix = ["Mr.", "Mrs.", "Ms.", "Miss.", "A.", "B.", "C.", "D.", "Dr.",
                   "sb.", "sth.", "n.", "v.", "vt.", "adj.", "prep.", "adv.", "conj.", "P.S"]
    """
    text = space_puncts(text)
    text = re.sub("(?<=\d)\. ", ".", text)
    text = re.sub("(?<=[A-D])\. ", ".", text)
    text = re.sub(" Mr \.| Mr\. ", " Mr.", text)
    text = re.sub(" Mrs \.| Mrs\. ", " Mrs.", text)
    text = re.sub(" Ms \.| Ms\. ", " Ms.", text)
    text = re.sub(" Miss \.| Miss\. ", " Miss.", text)
    text = re.sub(" Dr \.| Dr\. ", " Dr.", text)
    # n. v. "sb.", "sth.", "vt." 可能会造成原有该分句的，没被正确分句
    text = re.sub(" sb \. | sb\. ", " sb.", text)
    text = re.sub(" sth \. | sth\. ", " sth.", text)
    text = re.sub(" n \. | n\. ", " n.", text)
    text = re.sub(" v \. | v\. ", " v.", text)
    text = re.sub(" vt \. | vt\. ", " vt.", text)
    return text


def sent_preprocess(text):
    """
    text preprocessing after paragraph_tokenizer
    """
    text = space_puncts_1(text)
    for s in "{([<":
        text = text.replace(s, " " + s)
    for s in "})]>":
        text = text.replace(s, s + " ")
    text = text.replace("'s ", " 's ")
    text = text.replace("'re ", " 're ")
    text = text.replace("'ve ", " 've ")
    text = text.replace("'d ", " 'd ")
    text = text.replace("'m ", " 'm ")
    text = text.replace("'ll ", " 'll ")
    text = text.replace("n't ", " n't ")
    text = text.replace("can' t", "ca n't")
    text = re.sub("'s$", " 's", text)
    text = re.sub("'re$", " 're", text)
    text = re.sub("'ve$", " 've", text)
    text = re.sub("'d$", " 'd", text)
    text = re.sub("'m$", " 'm", text)
    text = re.sub("'ll$", " 'll", text)
    text = re.sub("n't$", " n't", text)
    text = re.sub("\.$", " .", text)
    text = re.sub(":$", " :", text)

    text = re.sub("(?<=\d)\.", ". ", text)
    text = re.sub("(?<=[A-D])\.", ". ", text)
    text = re.sub(" sb\.", " sb. ", text)
    text = re.sub(" sth\.", " sth. ", text)
    text = re.sub(" n\.", " n. ", text)
    text = re.sub(" v\.", " v. ", text)
    text = re.sub(" vt\.", " vt. ", text)
    return text


def special_token(context_string):
    """special token preprocessing and postion not change
    """
    table = {ord(f): ord(t) for f, t in zip(
        u'“”‘’′，、。．！？【】（）％＃＠＆１２３４５６７８９０',
        u'\"\"\'\'\',,..!?[]()%#@&1234567890')}
    text = context_string.translate(table)
    text = text.replace("…", "=")
    text = text.replace("—", "=")
    text = text.replace("#%%#", "####")
    return text


def text_preprocess(text):
    """
    :param text: 分句以及标点处理
    :return:
    """
    text = before_split_text(text)
    final_sents = []
    li_sents = paragraph_tokenizer.tokenize(text)
    for i in range(len(li_sents)):
        tmp_text = li_sents[i]
        for sent in re.split('\v|\f|\n', tmp_text):
            # 冒号前后未加空格
            final_sents.append(sent_preprocess(sent))
    return final_sents


def ratio_alphabetic(context_string):
    """
    :param context_string:
    :return: 返回英文所占字符串的比例
    """
    num = len(context_string)
    new_context_string = re.sub(r"[^a-zA-Z]", "", context_string)
    num_alphabetic = len(new_context_string)
    if num == 0:
        return 0
    ratio = num_alphabetic * 1.0 / num
    # 是否也可以认为是中文占比，0.4是否合理
    return ratio


def ratio_punct(context_string):
    """
    :param context_string:
    :return: 返回标点占单词的比例 >=1/3，则不语法纠错
    """
    words = context_string.split(" ")
    num = len(words)
    num_of_punct = 0
    punctuation = """,:;."""
    # punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    for word in words:
        if word in punctuation:
            num_of_punct += 1
    if num == 0:
        return 0
    ratio = num_of_punct * 1.0 / num
    return ratio


def remove_non_ascii(sents):
    replace_space_sents = []
    remove_space_sents = []
    for text in sents:
        text = text.replace("\t", " ").replace("\r", " ")
        for character in text:
            if ord(character) >= 127:
                text = text.replace(character, " ")
            if ord(character) <= 31:
                text = text.replace(character, " ")
        replace_space_sents.append(text)
        text = re.sub("\s{2,}", " ", text).strip()
        if text != "":
            remove_space_sents.append(text)
    return "#%%#".join(replace_space_sents), "#%%#".join(remove_space_sents)


def remove_nonascii_1(text):
    """
    用于保持请求中，句子的数量一致
    """
    text = text.replace("\t", " ").replace("\r", " ")
    for character in text:
        if ord(character) >= 127:
            text = text.replace(character, " ")
        if ord(character) <= 31:
            text = text.replace(character, " ")
    text = re.sub("\s{2,}", " ", text).strip()
    return text


def befroe_seq2seq_preprocess(text):
    """
    :param text:
    :return: the text input to seq2seq model
    """
    # 会显示的错误
    text = re.sub(" i ", ' I ', text)
    text = re.sub("^i ", 'I ', text)
    text = re.sub("''", '"', text)
    text = re.sub("``", '"', text)
    text = re.sub("`", "'", text)
    return text.strip()


def is_contain_no_ascii(text):
    """
    :param : 
    :return: 若文本中包含非ascii码字符，返回true，用于最后剔除修改
    """""
    for c in text:
        if ord(c) >= 127 or ord(c) <= 31:
            return True
    return False


def multi_same_tag(all_tags, words, num=3):
    """
    :param all_tags:
    :param words:
    :param num:
    :return:
    """
    count = 0
    for tag in all_tags:
        if tag in words:
            count += 1
        else:
            count = 0
        if count == num:
            return True

    if count >= num:
        return True
    else:
        return False


def not_gramamr_postag(text):
    """
    :param text:
    :return:
    """
    NN_words = ["NN", "NNS", "NNP", "NNPS"]
    VB_words = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    WH_words = ["WDT", "WP", "WP$", "WRB"]
    PREP_words = ["IN"]
    ADJ_words = ["JJ", "JJR", "JJS"]
    ADV_words = ["RB", "RBR", "RBS"]
    PRP_wors = ["PRP", "PRP$"]
    MD_words = ["MD"]
    CC_words = ["CC"]
    CD_words = ["CD"]
    DT_words = ["DT"]

    punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    all_tags = []
    words = text.split(" ")
    word_tags = nltk.pos_tag(words)
    # nltk的词性标注是上下文无关的
    for word_tag in word_tags:
        if word_tag[0] not in punctuation:
            all_tags.append(word_tag[1])
    NN_tag = multi_same_tag(all_tags, NN_words, num=4)
    if NN_tag:
        return NN_tag
    VB_tag = multi_same_tag(all_tags, VB_words, num=3)
    if VB_tag:
        return VB_tag
    WH_tag = multi_same_tag(all_tags, WH_words, num=3)
    if WH_tag:
        return WH_tag
    PREP_tag = multi_same_tag(all_tags, PREP_words, num=3)
    if PREP_tag:
        return PREP_tag
    ADJ_tag = multi_same_tag(all_tags, ADJ_words, num=3)
    if ADJ_tag:
        return ADJ_tag
    ADV_tag = multi_same_tag(all_tags, ADV_words, num=3)
    if ADV_tag:
        return ADV_tag
    PRP_tag = multi_same_tag(all_tags, PRP_wors, num=3)
    if PRP_tag:
        return PRP_tag
    MD_tag = multi_same_tag(all_tags, MD_words, num=3)
    if MD_tag:
        return MD_tag
    CC_tag = multi_same_tag(all_tags, CC_words, num=3)
    if CC_tag:
        return CC_tag
    CD_tag = multi_same_tag(all_tags, CD_words, num=3)
    if CD_tag:
        return CD_tag
    DT_tag = multi_same_tag(all_tags, DT_words, num=3)
    if DT_tag:
        return DT_tag
    return False


def not_gramamr_special_token(text):
    """
    :param text:
    :return:
    """
    grammar_tag = False
    special_token = [" sb.", " sth.", " n.", " v.", " vt.", " adj.", " prep.", " adv.", " conj.",
                     " sb ", " sth ", " n ", " v ", " vt ", " adj ", " prep ", " adv ", " conj ",
                     "A.", "B.", "C.", "D.",
                     "A , ", "B , ", "C , ", "D , ",
                     "A)", "B)", "C)", "D)",
                     "A>", "B>", "C>", "D>",
                     "=", "...", "——", "…",
                     "1: ", "2: ", "3: ", "4: ", "5: ", "6: ", "7: ", "8: ", "9: ", "0: ",
                     "1) ", "2) ", "3) ", "4) ", "5) ", "6) ", "7) ", "8) ", "9) ", "0) ",
                     "___", "&", " er ", " est ", " ly "]

    start_token = ["A, ", "B, ", "C, ", "D, ",
                   "1,", "2,", "3,", "4,", "5,", "6,", "7,", "8,", "9,",
                   "Unit 1", "Unit 2", "Unit 3", "Unit 4", "Unit 5",
                   "Unit 6", "Unit 7", "Unit 8", "Unit 9",
                   "unit 1", "unit 2", "unit 3", "unit 4", "unit 5", "unit 6", "unit 7", "unit 8", "unit 9"]
    for token in special_token:
        if token in text:
            grammar_tag = True
            break
    for token in start_token:
        if text.startswith(token):
            grammar_tag = True
            break
    # if not grammar_tag:
    #     is_yinbiao = re.search("/.+:/|\[.+:\]", text)
    #     if is_yinbiao != None:
    #         grammar_tag = True
    return grammar_tag


if __name__ == "__main__":
    text = "If you had not helped me , 1. I would have been drowned ."
    print(paragraph_tokenizer.tokenize(text))

    # import time
    #
    # t1 = time.time()
    # for i in range(1000):
    #     a = not_gramamr_postag(text)
    # print(time.time() - t1)
    #
    # t2 = time.time()
    # for i in range(1000):
    #     a = not_gramamr_special_token(text)
    # print(time.time() - t2)
    #
    # t3 = time.time()
    # for i in range(1000):
    #     a = ratio_punct(text)
    # print(time.time() - t3)
    #
    # t4 = time.time()
    # for i in range(1000):
    #     a = space_puncts_1(text)
    # print(time.time() - t4)
    # text = "We must esteem the ceremony and custom of every country ."
    # a = not_gramamr_special_token(text)
    # print(a)
