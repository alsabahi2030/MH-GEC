import argparse
import os
from difflib import SequenceMatcher

import Levenshtein
import numpy as np
from tqdm import tqdm
from utils.transform_suffixes import SuffixTransform, apply_suffixtransform
from utils.helpers import write_lines, read_parallel_lines, encode_verb_form, \
    apply_reverse_transformation, SEQ_DELIMETERS, START_TOKEN

def _split(token):
    if not token:
        return []
    parts = token.split()
    return parts or [token]


def apply_merge_transformation(source_tokens, target_words, shift_idx):
    edits = []
    if len(source_tokens) > 1 and len(target_words) == 1:
        # check merge
        transform = check_merge(source_tokens, target_words)
        if transform:
            for i in range(len(source_tokens) - 1):
                edits.append([(shift_idx + i, shift_idx + i + 1), transform])
            return edits

    if len(source_tokens) == len(target_words) == 2:
        # check swap
        transform = check_swap(source_tokens, target_words)
        if transform:
            edits.append([(shift_idx, shift_idx + 1), transform])
    return edits


def is_sent_ok(sent, delimeters=SEQ_DELIMETERS):
    for del_val in delimeters.values():
        if del_val in sent and del_val != " ":
            return False
    return True


def check_casetype(source_token, target_token,x=None):
    if source_token.lower() != target_token.lower():
        return None
    if source_token.lower() == target_token:
        return "$TRANSFORM_CASE_LOWER"
    elif source_token.capitalize() == target_token:
        return "$TRANSFORM_CASE_CAPITAL"
    elif source_token.upper() == target_token:
        return "$TRANSFORM_CASE_UPPER"
    elif source_token[1:].capitalize() == target_token[1:] and source_token[0] == target_token[0]:
        return "$TRANSFORM_CASE_CAPITAL_1"
    elif source_token[:-1].upper() == target_token[:-1] and source_token[-1] == target_token[-1]:
        return "$TRANSFORM_CASE_UPPER_-1"
    else:
        return None


def check_equal(source_token, target_token,x=None):
    if source_token == target_token:
        return "$KEEP"
    else:
        return None


def check_split(source_token, target_tokens):
    if source_token.split("-") == target_tokens:
        return "$TRANSFORM_SPLIT_HYPHEN"
    else:
        return None


def check_merge(source_tokens, target_tokens):
    if "".join(source_tokens) == "".join(target_tokens):
        return "$MERGE_SPACE"
    elif "-".join(source_tokens) == "-".join(target_tokens):
        return "$MERGE_HYPHEN"
    else:
        return None


def check_swap(source_tokens, target_tokens):
    if source_tokens == [x for x in reversed(target_tokens)]:
        return "$MERGE_SWAP"
    else:
        return None


def check_plural(source_token, target_token,x=None):
    if source_token.endswith("s") and source_token[:-1] == target_token:
        return "$TRANSFORM_AGREEMENT_SINGULAR"
    elif target_token.endswith("s") and source_token == target_token[:-1]:
        return "$TRANSFORM_AGREEMENT_PLURAL"
    else:
        return None


def check_verb(source_token, target_token,verb_form_dict):
    encoding = encode_verb_form(source_token, target_token,verb_form_dict)
    if encoding:
        return f"$TRANSFORM_VERB_{encoding}"
    else:
        return None


def apply_transformation(source_token, target_token,verb_form_dict):
    target_tokens = target_token.split()
    if len(target_tokens) > 1:
        # check split
        transform = check_split(source_token, target_tokens)
        if transform:
            return transform
    checks = [check_equal, check_casetype, check_verb, check_plural]
    for check in checks:
        transform = check(source_token, target_token,verb_form_dict)
        if transform:
            return transform
    transform = SuffixTransform(source_token, target_token).match()
    if transform:
        return transform
    return None

def replace_merge_transforms(tokens):
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens
    target_tokens = tokens[:]
    allowed_range = (1, len(tokens) - 1)
    for i in range(len(tokens)):
        target_token = tokens[i]
        if target_token.startswith("$MERGE"):
            if target_token.startswith("$MERGE_SWAP") and i in allowed_range:
                target_tokens[i - 1] = tokens[i + 1]
                target_tokens[i + 1] = tokens[i - 1]
                target_tokens[i: i + 1] = []
    target_line = " ".join(target_tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    return target_line.split()

#trans = apply_transformation('help','helped',)