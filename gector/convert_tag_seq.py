import logging
from typing import Dict, List
from allennlp.data.tokenizers import Token
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


PAD = "@@PADDING@@"
UNK = "@@UNKNOWN@@"
START_TOKEN = "$START"
SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR"}

delimeters: dict = SEQ_DELIMETERS
skip_correct: bool = False
skip_complex: int = 0
lazy: bool = False
max_len: int = None
test_mode: bool = False
tag_strategy: str = "keep_one"
tn_prob: float = 0
tp_prob: float = 0
broken_dot_strategy: str = "keep"
annotations =True

def write_lines(fn, lines, mode='w'):
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])

def read(file_path):
    # if `file_path` is a URL, redirect to the cache
    if annotations:
        ann_file_path = f"{file_path}.ann"
        mismatch_file_path = f"{file_path}.mis"

        with open(file_path, "r", encoding='utf8') as data_file, open(ann_file_path, "r",
                                                                      encoding='utf8') as ann_data_file, open(mismatch_file_path, "w", encoding='utf8') as mismatch_file:
            logger.info("Reading instances from lines in file at: %s and %s", file_path, ann_file_path)
            for line, line2 in zip(data_file, ann_data_file):
                line = line.strip("\n")
                line2 = line2.strip("\n")

                # skip blank and broken line
                tokens_and_tags1 = [pair.rsplit(delimeters['labels'], 1)
                                    for pair in line.split(delimeters['tokens'])]
                tokens_and_tags2 = [pair.rsplit(delimeters['labels'], 1)
                                    for pair in line2.split(delimeters['tokens'])]
                try:
                    tokens1 = [Token(token) for token, tag in tokens_and_tags1]
                    tokens2 = [Token(token) for token, tag in tokens_and_tags2]

                    tags1 = [tag for token, tag in tokens_and_tags1]
                    tags2 = [tag for token, tag in tokens_and_tags2]

                except ValueError:
                    tokens1 = [Token(token[0]) for token in tokens_and_tags1]
                    tokens2 = [Token(token[0]) for token in tokens_and_tags2]

                    tags1 = None
                    tags2 = None

                if tokens1 and tokens1[0] != Token(START_TOKEN):
                    tokens1 = [Token(START_TOKEN)] + tokens1
                if tokens2 and tokens2[0] != Token(START_TOKEN):
                    tokens2 = [Token(START_TOKEN)] + tokens2

                words1 = [x.text for x in tokens1]
                words2 = [x.text for x in tokens2]
                m_tag1 = []
                m_tag2 = []
                for t in tags1:
                    if t != "$KEEP":
                        m_tag1.append("XXXX")
                    else:
                        m_tag1.append(t)
                for t2 in tags2:
                    if t2 != "$KEEP":
                        m_tag2.append("XXXX")
                    else:
                        m_tag2.append(t2)


                if m_tag1 != m_tag2:
                    logger.info(f"words mismatch, {m_tag1} , {m_tag2}")
                    #m_tag1.extend(m_tag2)
                    word1 = ' '.join(words1) + '\n'
                    mismatch_file.write(word1)
                    tag1 = ' '.join(tags1) + '\n'
                    mismatch_file.write(tag1)
                    word2 = ' '.join(words2) + '\n'
                    mismatch_file.write(word2)
                    tag2 = ' '.join(tags2) + '\n'
                    mismatch_file.write(tag2)
                    mismatch_file.write('\n')

                    continue

                if words1 != words2:
                    logger.info(f"words mismatch, {words1} , {words2}")
                    continue
"""
                instance = self.text_to_instance(tokens1, tags1, tags2, words1)

                if instance:
                    yield instance
    else:
        with open(file_path, "r", encoding='utf8') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                # skip blank and broken lines
                if not line or (not test_mode and broken_dot_strategy == 'skip'
                                and BROKEN_SENTENCES_REGEXP.search(line) is not None):
                    continue

                tokens_and_tags = [pair.rsplit(delimeters['labels'], 1)
                                   for pair in line.split(delimeters['tokens'])]
                try:
                    tokens = [Token(token) for token, tag in tokens_and_tags]
                    tags = [tag for token, tag in tokens_and_tags]
                except ValueError:
                    tokens = [Token(token[0]) for token in tokens_and_tags]
                    tags = None

                if tokens and tokens[0] != Token(START_TOKEN):
                    tokens = [Token(START_TOKEN)] + tokens

                words = [x.text for x in tokens]
                if max_len is not None:
                    tokens = tokens[:max_len]
                    tags = None if tags is None else tags[:max_len]
                instance = self.text_to_instance(tokens, tags, words)
                if instance:
                    yield instance
"""

def extract_tags(self, tags: List[str], tags2: List[str] = None):
    op_del = self._delimeters['operations']

    labels = [x.split(op_del) for x in tags]
    if self._annotations:
        labels2 = [x.split(op_del) for x in tags2]

    comlex_flag_dict = {}
    # get flags
    for i in range(5):
        idx = i + 1
        comlex_flag_dict[idx] = sum([len(x) > idx for x in labels])

    if self._tag_strategy == "keep_one":
        # get only first candidates for r_tags in right and the last for left
        labels = [x[0] for x in labels]
        if self._annotations:
            labels2 = [x[0] for x in labels2]

    elif self._tag_strategy == "merge_all":
        # consider phrases as a words
        pass
    else:
        raise Exception("Incorrect tag strategy")

    detect_tags = ["CORRECT" if label == "$KEEP" else "INCORRECT" for label in labels]
    if self._multiclassifier or self._early_exit1:
        delete_tags = ["DELETE" if label.startswith("$DELETE") else "KEEP" for label in labels]
        replace_tags = ["REPLACE" if label.startswith("$REPLACE") else "KEEP" for label in labels]
        append_tags = ["INSERT" if label.startswith("$APPEND") else "KEEP" for label in labels]
        if self._multiclassifier2 or self._early_exit2:
            transform_tags = ["TRANSFORM" if 'TRANSFORM' in label else "KEEP" for label in labels]
            merge_tags = ["MERGE" if label.startswith("$MERGE") else "KEEP" for label in labels]
            return labels, detect_tags, delete_tags, replace_tags, append_tags, transform_tags, merge_tags, comlex_flag_dict

        return labels, detect_tags, delete_tags, replace_tags, append_tags, comlex_flag_dict
    if self._annotations:
        return labels, labels2, detect_tags, comlex_flag_dict

    return labels, detect_tags, comlex_flag_dict

read("/data_local/src/gector-master/new_data/temp/wi.dev.tagged")