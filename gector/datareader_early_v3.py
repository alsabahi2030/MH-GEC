"""Tweaked AllenNLP dataset reader."""
import logging
import re
from random import random
from typing import Dict, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from utils.helpers import SEQ_DELIMETERS, START_TOKEN

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("seq2labels_datareader")
class Seq2LabelsDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    delimiters: ``dict``
        The dcitionary with all delimeters.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    max_len: if set than will truncate long sentences
    """
    # fix broken sentences mostly in Lang8
    BROKEN_SENTENCES_REGEXP = re.compile(r'\.[a-zA-RT-Z]')

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimeters: dict = SEQ_DELIMETERS,
                 skip_correct: bool = False,
                 skip_complex: int = 0,
                 lazy: bool = False,
                 max_len: int = None,
                 test_mode: bool = False,
                 tag_strategy: str = "keep_one",
                 tn_prob: float = 0,
                 tp_prob: float = 0,
                 broken_dot_strategy: str = "keep",
                 multiclassifier: bool = False, multiclassifier2: bool = False, early_exit1: bool = False, early_exit2: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimeters = delimeters
        self._max_len = max_len
        self._skip_correct = skip_correct
        self._skip_complex = skip_complex
        self._tag_strategy = tag_strategy
        self._broken_dot_strategy = broken_dot_strategy
        self._test_mode = test_mode
        self._tn_prob = tn_prob
        self._tp_prob = tp_prob
        self._multiclassifier = multiclassifier
        self._multiclassifier2 = multiclassifier2
        self._early_exit1 = early_exit1
        self._early_exit2 = early_exit2

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r", encoding='utf8') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                # skip blank and broken lines
                if not line or (not self._test_mode and self._broken_dot_strategy == 'skip'
                                and self.BROKEN_SENTENCES_REGEXP.search(line) is not None):
                    continue

                tokens_and_tags = [pair.rsplit(self._delimeters['labels'], 1)
                                   for pair in line.split(self._delimeters['tokens'])]
                try:
                    tokens = [Token(token) for token, tag in tokens_and_tags]
                    tags = [tag for token, tag in tokens_and_tags]
                except ValueError:
                    tokens = [Token(token[0]) for token in tokens_and_tags]
                    tags = None

                if tokens and tokens[0] != Token(START_TOKEN):
                    tokens = [Token(START_TOKEN)] + tokens

                words = [x.text for x in tokens]
                if self._max_len is not None:
                    tokens = tokens[:self._max_len]
                    tags = None if tags is None else tags[:self._max_len]
                instance = self.text_to_instance(tokens, tags, words)
                if instance:
                    yield instance

    def extract_tags(self, tags: List[str],tags2: List[str]=None):
        op_del = self._delimeters['operations']

        labels = [x.split(op_del) for x in tags]

        comlex_flag_dict = {}
        # get flags
        for i in range(5):
            idx = i + 1
            comlex_flag_dict[idx] = sum([len(x) > idx for x in labels])

        if self._tag_strategy == "keep_one":
            # get only first candidates for r_tags in right and the last for left
            labels = [x[0] for x in labels]

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
            if self._multiclassifier2  or self._early_exit2:
                transform_tags = ["TRANSFORM" if 'TRANSFORM' in  label else "KEEP" for label in labels]
                merge_tags = ["MERGE" if label.startswith("$MERGE") else "KEEP" for label in labels]
                return labels, detect_tags, delete_tags, replace_tags, append_tags, transform_tags,merge_tags, comlex_flag_dict

            return labels, detect_tags, delete_tags, replace_tags, append_tags, comlex_flag_dict


        return labels, detect_tags, comlex_flag_dict

    def text_to_instance(self, tokens: List[Token], tags1: List[str] = None, tags2: List[str] = None,
                         words: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": words})
        if tags1 is not None:
            if self._multiclassifier2 or self._early_exit2:
                labels, detect_tags, delete_tags, replace_tags, append_tags, transform_tags,merge_tags, complex_flag_dict = self.extract_tags(tags1)
            elif self._multiclassifier or self._early_exit1:
                labels, detect_tags, delete_tags, replace_tags, append_tags, complex_flag_dict = self.extract_tags(tags1)
            else:
                labels, detect_tags, complex_flag_dict = self.extract_tags(tags1)
            if self._skip_complex and complex_flag_dict[self._skip_complex] > 0:
                return None
            rnd = random()
            # skip TN
            if self._skip_correct and all(x == "CORRECT" for x in detect_tags):
                if rnd > self._tn_prob:
                    return None
            # skip TP
            else:
                if rnd > self._tp_prob:
                    return None

            fields["labels"] = SequenceLabelField(labels, sequence,
                                                  label_namespace="labels")
            fields["d_tags"] = SequenceLabelField(detect_tags, sequence,
                                                  label_namespace="d_tags")
            if self._multiclassifier  or self._early_exit1:
                fields["del_tags"] = SequenceLabelField(delete_tags, sequence,
                                                      label_namespace="del_tags")
                fields["rep_tags"] = SequenceLabelField(replace_tags, sequence,
                                                      label_namespace="rep_tags")

                fields["app_tags"] = SequenceLabelField(append_tags, sequence,
                                                        label_namespace="app_tags")
                if self._multiclassifier2 or self._early_exit2:
                    fields["trans_tags"] = SequenceLabelField(transform_tags, sequence,
                                                            label_namespace="trans_tags")
                    fields["merge_tags"] = SequenceLabelField(merge_tags, sequence,
                                                            label_namespace="merge_tags")

        return Instance(fields)