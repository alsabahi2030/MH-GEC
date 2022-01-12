"""Basic model. Predicts tags for every token"""
from typing import Dict, Optional, List, Any

import numpy
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.nn import util
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch.nn.modules.linear import Linear


@Model.register("seq2labels")
class Seq2Labels(Model):
    """
    This ``Seq2Labels`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag (or couple tags) for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1.
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` is true.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric, if desired.
        Unless you did something unusual, the default value should be what you want.
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 predictor_dropout=0.0,
                 labels_namespace: str = "labels",
                 detect_namespace: str = "d_tags",
                 delete_namespace: str = "del_tags",
                 replace_namespace: str = "rep_tags",
                 append_namespace: str = "app_tags",
                 transform_namespace: str = "trans_tags",
                 merge_namespace: str = "merge_tags",
                 ann_labels_namespace: str = "ann_labels",

                 verbose_metrics: bool = False,
                 label_smoothing: float = 0.0,
                 confidence: float = 0.0,
                 twolayers_classifier: bool = False,
                 relu_activation: bool = False,
                 multiclassifier: bool = False,
                 multiclassifier2: bool = False,
                 early_exit1: bool = False,
                 early_exit2: bool = False,
                 multiheadversion: int = 1,
                 lamda:float = 1,
                 predictonly: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,use_cpu: bool = False) -> None:
        super(Seq2Labels, self).__init__(vocab, regularizer)
        self.use_cpu = use_cpu

        self.label_namespaces = [labels_namespace, ann_labels_namespace,
                                 detect_namespace, delete_namespace, replace_namespace, append_namespace, transform_namespace,merge_namespace]
        self.text_field_embedder = text_field_embedder
        self.num_labels_classes = self.vocab.get_vocab_size(labels_namespace)

        self.num_detect_classes = self.vocab.get_vocab_size(detect_namespace)

        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.twolayers_classifier = twolayers_classifier
        self.relu_activation = relu_activation
        self.multiclassifier = multiclassifier
        self.multiclassifier2 = multiclassifier2
        self.early_exit1 = early_exit1
        self.early_exit2 = early_exit2
        self.multiheadversion = multiheadversion


        self.lamda = lamda
        self.predictonly = predictonly
        self.incorr_index = self.vocab.get_token_index("INCORRECT",
                                                       namespace=detect_namespace)
        self.corr_index = self.vocab.get_token_index("CORRECT",
                                                       namespace=detect_namespace)
        if not predictonly and (self.multiclassifier or self.early_exit1):
            self.num_delete_classes = self.vocab.get_vocab_size(delete_namespace)
            self.num_replace_classes = self.vocab.get_vocab_size(replace_namespace)
            self.num_append_classes = self.vocab.get_vocab_size(append_namespace)

            self.del_index = self.vocab.get_token_index("DELETE",
                                                           namespace=delete_namespace)
            self.rep_index = self.vocab.get_token_index("REPLACE",
                                                           namespace=replace_namespace)
            self.app_index = self.vocab.get_token_index("APPEND",
                                                               namespace=append_namespace)
            if self.multiclassifier2 or self.early_exit2:
                self.num_transform_classes = self.vocab.get_vocab_size(transform_namespace)
                self.num_merge_classes = self.vocab.get_vocab_size(merge_namespace)

                self.trans_index = self.vocab.get_token_index("TRANSFORM",
                                                                   namespace=transform_namespace)
                self.merge_index = self.vocab.get_token_index("MERGE",
                                                                   namespace=merge_namespace)
        self._verbose_metrics = verbose_metrics
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(predictor_dropout))
        if not self.twolayers_classifier:
            self.tag_detect_projection_layer = TimeDistributed(
                Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_detect_classes))
            if self.multiclassifier and not self.predictonly:
                self.tag_delete_projection_layer = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_delete_classes))

                self.tag_replace_projection_layer = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_replace_classes))

                self.tag_append_projection_layer = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_append_classes))
                if self.multiclassifier2:
                    self.tag_transform_projection_layer = TimeDistributed(
                        Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_transform_classes))
                    self.tag_merge_projection_layer = TimeDistributed(
                        Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_merge_classes))
                self.tag_labels_projection_layer = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_labels_classes))
            elif self.early_exit1:
                self.tag_labels_projection_layer0 = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(),
                           text_field_embedder._token_embedders['bert'].get_output_dim()))
                self.tag_labels_projection_layer1 = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(),
                           text_field_embedder._token_embedders['bert'].get_output_dim()))

                self.tag_labels_projection_layer2 = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(),
                           text_field_embedder._token_embedders['bert'].get_output_dim()))

                self.tag_labels_projection_layer3 = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(),
                           text_field_embedder._token_embedders['bert'].get_output_dim()))
                if self.early_exit2:
                    self.tag_labels_projection_layer4 = TimeDistributed(
                        Linear(text_field_embedder._token_embedders['bert'].get_output_dim(),
                               text_field_embedder._token_embedders['bert'].get_output_dim()))

                    self.tag_labels_projection_layer5 = TimeDistributed(
                        Linear(text_field_embedder._token_embedders['bert'].get_output_dim(),
                               text_field_embedder._token_embedders['bert'].get_output_dim()))

                self.tag_labels_projection_layer6 = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_labels_classes))

                self.tag_delete_projection_layer = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_delete_classes))
                self.tag_replace_projection_layer = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_replace_classes))
                self.tag_append_projection_layer = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_append_classes))
                if self.early_exit2:
                    self.tag_transform_projection_layer = TimeDistributed(
                        Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_transform_classes))
                    self.tag_merge_projection_layer = TimeDistributed(
                        Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_merge_classes))

            else:
                self.tag_labels_projection_layer = TimeDistributed(
                    Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_labels_classes))
        else:
            self.tag_labels_projection_layer1 = TimeDistributed(
                Linear(text_field_embedder._token_embedders['bert'].get_output_dim(),
                       text_field_embedder._token_embedders['bert'].get_output_dim()))

            self.tag_detect_projection_layer1 = TimeDistributed(
                Linear(text_field_embedder._token_embedders['bert'].get_output_dim(),
                       text_field_embedder._token_embedders['bert'].get_output_dim()))

            self.tag_labels_projection_layer2 = TimeDistributed(
                Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_labels_classes))

            self.tag_detect_projection_layer2 = TimeDistributed(
                Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_detect_classes))
        self.metrics = {"accuracy": CategoricalAccuracy()}

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                d_tags: torch.LongTensor = None,
                del_tags: torch.LongTensor = None,
                app_tags: torch.LongTensor = None,
                rep_tags: torch.LongTensor = None,
                trans_tags: torch.LongTensor = None,
                merge_tags: torch.LongTensor = None,
                ann_labels: torch.LongTensor = None,

                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        lables : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        encoded_text = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = encoded_text.size()
        mask = get_text_field_mask(tokens)
        if not self.twolayers_classifier:
            #logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(encoded_text))
            if self.multiclassifier  and not self.predictonly:
                logits_d = self.tag_detect_projection_layer(encoded_text)
                logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(encoded_text))
                logits_del = self.tag_delete_projection_layer(encoded_text)
                logits_rep = self.tag_replace_projection_layer(encoded_text)
                logits_app = self.tag_append_projection_layer(encoded_text)
                if self.multiclassifier2:
                    logits_trans = self.tag_transform_projection_layer(encoded_text)
                    logits_merge = self.tag_merge_projection_layer(encoded_text)
            elif self.early_exit2  and not self.predictonly:
                if self.multiheadversion == 1:
                    logits_d = self.tag_detect_projection_layer(encoded_text)
                    logits_labels_del = self.tag_labels_projection_layer1(self.predictor_dropout(encoded_text))
                    logits_labels_rep = self.tag_labels_projection_layer2(logits_labels_del)
                    logits_labels_app = self.tag_labels_projection_layer3(logits_labels_rep)
                    logits_labels_trans = self.tag_labels_projection_layer4(logits_labels_app)
                    logits_labels_merge = self.tag_labels_projection_layer5(logits_labels_trans)
                    logits_labels = self.tag_labels_projection_layer6(logits_labels_merge)

                elif self.multiheadversion == 2:
                    logits_d = self.tag_detect_projection_layer(encoded_text)
                    logits_labels_del = self.tag_labels_projection_layer1(self.predictor_dropout(encoded_text))
                    logits_labels_merge = self.tag_labels_projection_layer5(logits_labels_del)
                    logits_labels_trans = self.tag_labels_projection_layer4(logits_labels_merge)
                    logits_labels_app = self.tag_labels_projection_layer3(logits_labels_trans)
                    logits_labels_rep = self.tag_labels_projection_layer2(logits_labels_app)
                    logits_labels = self.tag_labels_projection_layer6(logits_labels_rep)
                elif self.multiheadversion == 3:
                    logits_labels_det = self.tag_labels_projection_layer0(self.predictor_dropout(encoded_text))
                    logits_labels_del = self.tag_labels_projection_layer1(logits_labels_det)
                    logits_labels_merge = self.tag_labels_projection_layer5(logits_labels_del)
                    logits_labels_trans = self.tag_labels_projection_layer4(logits_labels_merge)
                    logits_labels_app = self.tag_labels_projection_layer3(logits_labels_trans)
                    logits_labels_rep = self.tag_labels_projection_layer2(logits_labels_app)
                    logits_labels = self.tag_labels_projection_layer6(logits_labels_rep)
                    logits_d = self.tag_detect_projection_layer(logits_labels_det)
                elif self.multiheadversion == 4:
                    #logits_labels_det = self.tag_labels_projection_layer0(self.predictor_dropout(encoded_text))
                    logits_labels_del = self.tag_labels_projection_layer1(self.predictor_dropout(encoded_text))
                    logits_labels_merge = self.tag_labels_projection_layer5(logits_labels_del)
                    logits_labels_trans = self.tag_labels_projection_layer4(logits_labels_merge)
                    logits_labels_app = self.tag_labels_projection_layer3(logits_labels_trans)
                    logits_labels_rep = self.tag_labels_projection_layer2(logits_labels_app)
                    logits_labels = self.tag_labels_projection_layer6(logits_labels_rep)
                    logits_d = self.tag_detect_projection_layer(logits_labels_rep)


                logits_del = self.tag_delete_projection_layer(logits_labels_del)
                logits_rep = self.tag_replace_projection_layer(logits_labels_rep)
                logits_app = self.tag_append_projection_layer(logits_labels_app)
                logits_trans = self.tag_transform_projection_layer(logits_labels_trans)
                logits_merge = self.tag_merge_projection_layer(logits_labels_merge)

            elif self.early_exit1  and not self.predictonly:
                logits_d = self.tag_detect_projection_layer(encoded_text)
                logits_labels_del = self.tag_labels_projection_layer1(self.predictor_dropout(encoded_text))
                logits_labels_rep = self.tag_labels_projection_layer2(logits_labels_del)
                logits_labels_app = self.tag_labels_projection_layer3(logits_labels_rep)
                logits_labels = self.tag_labels_projection_layer6(logits_labels_app)

                logits_del = self.tag_delete_projection_layer(logits_labels_del)
                logits_rep = self.tag_replace_projection_layer(logits_labels_rep)
                logits_app = self.tag_append_projection_layer(logits_labels_app)


            else:
                logits_d = self.tag_detect_projection_layer(encoded_text)
                logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(encoded_text))

        elif self.relu_activation:
            pre_logits_labels = F.relu(self.tag_labels_projection_layer1(self.predictor_dropout(encoded_text)))
            pre_logits_d = F.relu(self.tag_detect_projection_layer1(encoded_text))

            logits_labels = self.tag_labels_projection_layer2(pre_logits_labels)
            logits_d = self.tag_detect_projection_layer2(pre_logits_d)
        else:
            pre_logits_labels = self.tag_labels_projection_layer1(self.predictor_dropout(encoded_text))
            pre_logits_d = self.tag_detect_projection_layer1(encoded_text)

            logits_labels = self.tag_labels_projection_layer2(pre_logits_labels)
            logits_d = self.tag_detect_projection_layer2(pre_logits_d)

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes])


        class_probabilities_d = F.softmax(logits_d, dim=-1).view(
            [batch_size, sequence_length, self.num_detect_classes])
        error_probs_det = class_probabilities_d[:, :, self.incorr_index] * mask
        corr_probs_det = class_probabilities_d[:, :, self.corr_index] * mask

        incorr_prob = torch.max(error_probs_det, dim=-1)[0]

        if  not self.predictonly and (self.multiclassifier or self.early_exit1):
            class_probabilities_del = F.softmax(logits_del, dim=-1).view(
                [batch_size, sequence_length, self.num_delete_classes])
            class_probabilities_rep = F.softmax(logits_rep, dim=-1).view(
                [batch_size, sequence_length, self.num_replace_classes])
            class_probabilities_app = F.softmax(logits_app, dim=-1).view(
                [batch_size, sequence_length, self.num_append_classes])

            error_probs_del = class_probabilities_del[:, :, self.del_index] * mask
            error_probs_rep = class_probabilities_rep[:, :, self.rep_index] * mask
            error_probs_app = class_probabilities_app[:, :, self.app_index] * mask
            if self.multiclassifier2 or self.early_exit2:
                class_probabilities_trans = F.softmax(logits_trans, dim=-1).view(
                    [batch_size, sequence_length, self.num_transform_classes])
                class_probabilities_merge = F.softmax(logits_merge, dim=-1).view(
                    [batch_size, sequence_length, self.num_merge_classes])
                error_probs_trans = class_probabilities_trans[:, :, self.trans_index] * mask
                error_probs_merge = class_probabilities_merge[:, :, self.merge_index] * mask


            #del_prob = torch.max(error_probs_del, dim=-1)[0]
            #rep_prob = torch.max(error_probs_rep, dim=-1)[0]
            #app_prob = torch.max(error_probs_app, dim=-1)[0]

        if self.confidence > 0:
            probability_change = [self.confidence] + [0] * (self.num_labels_classes - 1)
            if self.use_cpu:
                class_probabilities_labels += torch.FloatTensor(probability_change).repeat(
                (batch_size, sequence_length, 1))
            else:
                class_probabilities_labels += torch.cuda.FloatTensor(probability_change).repeat(
                (batch_size, sequence_length, 1))

        if  not self.predictonly and (self.multiclassifier2 or self.early_exit2):
            output_dict = {"logits_labels": logits_labels,
                           "logits_d_tags": logits_d,
                           "logits_del_tags": logits_del,
                           "logits_rep_tags": logits_rep,
                           "logits_app_tags": logits_app,
                           "logits_trans_tags": logits_trans,
                           "logits_merge_tags": logits_merge,

                           "class_probabilities_labels": class_probabilities_labels,
                           "class_probabilities_d_tags": class_probabilities_d,
                           "class_probabilities_del_tags": class_probabilities_del,
                           "class_probabilities_rep_tags": class_probabilities_rep,
                           "class_probabilities_app_tags": class_probabilities_app,
                           "class_probabilities_trans_tags": class_probabilities_trans,
                           "class_probabilities_merge_tags": class_probabilities_merge,
                           "max_error_probability": incorr_prob}
        elif  not self.predictonly and (self.multiclassifier or self.early_exit1 ):
            output_dict = {"logits_labels": logits_labels,
                           "logits_d_tags": logits_d,
                           "logits_del_tags": logits_del,
                           "logits_rep_tags": logits_rep,
                           "logits_app_tags": logits_app,

                           "class_probabilities_labels": class_probabilities_labels,
                           "class_probabilities_d_tags": class_probabilities_d,
                           "class_probabilities_del_tags": class_probabilities_del,
                           "class_probabilities_rep_tags": class_probabilities_rep,
                           "class_probabilities_app_tags": class_probabilities_app,
                           "max_error_probability": incorr_prob}

        else:
            output_dict = {"logits_labels": logits_labels,
                       "logits_d_tags": logits_d,
                       "class_probabilities_labels": class_probabilities_labels,
                       "class_probabilities_d_tags": class_probabilities_d,
                       "max_error_probability": incorr_prob}

        if self.multiclassifier or self.early_exit1:
            if labels is not None and d_tags is not None and del_tags is not None and rep_tags is not None and app_tags is not None :
                loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask,
                                                                 label_smoothing=self.label_smoothing)
                loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
                loss_del = sequence_cross_entropy_with_logits(logits_del, del_tags, mask)
                loss_rep = sequence_cross_entropy_with_logits(logits_rep, rep_tags, mask)
                loss_app = sequence_cross_entropy_with_logits(logits_app, app_tags, mask)
                if (self.multiclassifier2  or self.early_exit2) and trans_tags is not None and merge_tags is not None:
                    loss_trans = sequence_cross_entropy_with_logits(logits_trans, trans_tags, mask)
                    loss_merge = sequence_cross_entropy_with_logits(logits_merge, trans_tags, mask)

                for metric in self.metrics.values():
                    metric(logits_labels, labels, mask.float())
                    metric(logits_d, d_tags, mask.float())
                    metric(logits_del, del_tags, mask.float())
                    metric(logits_rep, rep_tags, mask.float())
                    metric(logits_app, app_tags, mask.float())
                    if self.multiclassifier2  or self.early_exit2:
                        metric(logits_trans, trans_tags, mask.float())
                        metric(logits_merge, merge_tags, mask.float())
                if (self.multiclassifier2  or self.early_exit2) and trans_tags is not None and merge_tags is not None:
                    output_dict["loss"] = loss_labels + loss_d  + loss_del * self.lamda + loss_rep * self.lamda + loss_app * self.lamda + loss_trans * self.lamda + loss_merge * self.lamda

                else:
                    output_dict["loss"] = loss_labels + loss_d + loss_del * self.lamda + loss_rep * self.lamda + loss_app * self.lamda

        elif labels is not None and d_tags is not None:
            loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask,
                                                             label_smoothing=self.label_smoothing)
            loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
            for metric in self.metrics.values():
                metric(logits_labels, labels, mask.float())
                metric(logits_d, d_tags, mask.float())
            output_dict["loss"] = loss_labels + loss_d

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        for label_namespace in self.label_namespaces:
            all_predictions = output_dict[f'class_probabilities_{label_namespace}']
            all_predictions = all_predictions.cpu().data.numpy()
            if all_predictions.ndim == 3:
                predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
            else:
                predictions_list = [all_predictions]
            all_tags = []

            for predictions in predictions_list:
                argmax_indices = numpy.argmax(predictions, axis=-1)
                tags = [self.vocab.get_token_from_index(x, namespace=label_namespace)
                        for x in argmax_indices]
                all_tags.append(tags)
            output_dict[f'{label_namespace}'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
        return metrics_to_return
