"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
import logging
import os
import sys
from time import time

import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.seq2labels_model_earlyexit_v6 import Seq2Labels
from gector.wordpiece_indexer import PretrainedBertIndexer
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)

def get_weights_name(transformer_name, lowercase):
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    if transformer_name == 'distilbert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'distilbert-base-uncased'
    if transformer_name == 'albert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        print('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'
    if transformer_name == 'xlnet-large':
        return 'xlnet-large-cased'
    if transformer_name == 'roberta-large':
        return 'roberta-large'


class GecBERTModel(object):
    def __init__(self, vocab_path=None, model_paths=None,
                 weigths=None,
                 max_len=50,
                 min_len=3,
                 lowercase_tokens=False,
                 log=False,
                 iterations=3,
                 min_probability=0.0,
                 model_name='roberta',
                 special_tokens_fix=1,
                 is_ensemble=True,
                 min_error_probability=0.0,
                 confidence=0,
                 twolaysers_classifier=False,
                 multiclassifier=False,
                 multiclassifier2=False,
                 early_exit1=False,
                 early_exit2=False,
                 multiheadversion=1,
                 cfs=None,
                 lamda=1,
                 predictonly=False,
                 relu_activation=False,
                 resolve_cycles=False,
                 use_cpu=False
                 ):
        self.use_cpu = use_cpu
        self.model_weights = list(map(float, weigths)) if weigths else [1] * len(model_paths)
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_cpu == False else "cpu")

        self.max_len = max_len
        self.min_len = min_len
        self.lowercase_tokens = lowercase_tokens
        #self.min_probability = min_probability
        self.min_probability = min_error_probability
        """
        if mps and cfs:
            self.mps_cfs = [(float(m),float(c)) for m,c in zip(mps,cfs)]
        else:
            self.mps_cfs =None
        """
        if cfs is not None:
            self.cfs = [float(c) for c in cfs]
        else:
            self.cfs = None
        self.min_error_probability = min_error_probability
        if vocab_path:
            self.vocab = Vocabulary.from_files(vocab_path)
        self.log = log
        self.iterations = iterations
        self.confidence = confidence
        self.twolaysers_classifier = twolaysers_classifier
        self.multiclassifier = multiclassifier
        self.multiclassifier2 = multiclassifier2
        self.early_exit1 = early_exit1
        self.early_exit2 = early_exit2
        self.multiheadversion = multiheadversion

        self.is_ensemble = is_ensemble
        self.lamda = lamda
        self.predictonly = predictonly

        self.relu_activation = relu_activation
        self.resolve_cycles = resolve_cycles
        # set training parameters and operations

        self.indexers = []
        self.models = []
        for idx, model_path in enumerate(model_paths):
            if is_ensemble:
                model_name, special_tokens_fix, self.twolaysers_classifier, self.multiclassifier, self.multiclassifier2, self.annotations, self.vocab_path, self.early_exit1, self.early_exit2, self.multiheadversion = self._get_model_data(model_path)
                self.vocab = Vocabulary.from_files(self.vocab_path)
            weights_name = get_weights_name(model_name, lowercase_tokens)
            if self.cfs is not None:
                self.confidence = self.cfs[idx]
            #weights_name ="/data/kamal/src/gector-master/pretrained_models/xlnet_cased_L-12_H-768_A-12/"
            self.indexers.append(self._get_indexer(weights_name, special_tokens_fix))

            model = Seq2Labels(vocab=self.vocab,
                               text_field_embedder=self._get_embbeder(weights_name, special_tokens_fix),
                               confidence=self.confidence,
                               twolayers_classifier=self.twolaysers_classifier,
                               relu_activation=self.relu_activation,
                               multiclassifier=self.multiclassifier,
                               multiclassifier2=self.multiclassifier2,
                               early_exit1=self.early_exit1,
                               early_exit2=self.early_exit2,
                               multiheadversion=self.multiheadversion,
                               lamda=self.lamda, predictonly=self.predictonly

                               ).to(self.device)
            if torch.cuda.is_available():
                if self.predictonly:
                    strict = False
                else:
                    strict =True
                model.load_state_dict(torch.load(model_path),strict=strict)
            else:
                model.load_state_dict(torch.load(model_path,
                                                 map_location=torch.device('cpu')))
            model.eval()
            self.models.append(model)

    @staticmethod
    def _get_model_data(model_path):
        model_name = model_path.split('/')[-1]
        tr_model, stf, twol = model_name.split('_')[:3]
        twolclassifier = False
        multiclassifier = False
        multiclassifier2 = False
        early_exit1 = False
        early_exit2 = False
        multiheadversion = 1

        vocal_path = "./data/output_suffix5k_vocabulary/"
        annotations = False
        if 'ann' in twol:
            annotations = True

        if '2l' in twol:
            twolclassifier =True

        if 'mc2' in twol:
            multiclassifier = True
            multiclassifier2 = True
            vocal_path = './data/output_suffix5kordered_vocabulary/'

        elif 'mc1' in twol:
            multiclassifier = True
            vocal_path = './data/output_suffix5kordered_vocabulary/'
        elif 'ex2' in twol:
            vocal_path = './data/output_suffix5kordered_vocabulary/'
            early_exit1= True
            early_exit2= True
            if 'v2' in twol:
                multiheadversion = 2
        elif 'ex1' in twol:
            vocal_path = './data/output_suffix5kordered_vocabulary/'
            early_exit1= True
        if 'newann' in twol:
            vocal_path = './data/output_newtagnewannsuffix5korderedwifcnlclean_vocabulary/'
        elif 'newtag' in twol:
            vocal_path = './data/output_newtagsuffix5korderedwifcnlclean_vocabulary/'
        return tr_model, int(stf), twolclassifier, multiclassifier, multiclassifier2, annotations,vocal_path,early_exit1,early_exit2,multiheadversion

    def _restore_model(self, input_path):
        if os.path.isdir(input_path):
            print("Model could not be restored from directory", file=sys.stderr)
            filenames = []
        else:
            filenames = [input_path]

        for model_path in filenames:
            try:
                if torch.cuda.is_available():
                    loaded_model = torch.load(model_path)
                else:
                    loaded_model = torch.load(model_path,
                                              map_location=lambda storage,
                                                                  loc: storage)
            except:
                print(f"{model_path} is not valid model", file=sys.stderr)
            own_state = self.model.state_dict()

            for name, weights in loaded_model.items():
                if name not in own_state:
                    continue
                try:
                    if len(filenames) == 1:
                        own_state[name].copy_(weights)
                    else:
                        own_state[name] += weights
                except RuntimeError:
                    continue
        print("Model is restored", file=sys.stderr)

    def predict(self, batches):
        t11 = time()
        predictions = []

        for batch, model in zip(batches, self.models):
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
            with torch.no_grad():
                prediction = model.forward(**batch)
            predictions.append(prediction)

        preds, idx, error_probs, ann_preds, ann_idx  = self._convert(predictions)
        t55 = time()

        if self.log:
            print(f"Inference time {t55 - t11}")

        return preds, idx, error_probs, ann_preds, ann_idx

    def get_token_action(self, token, index, prob, sugg_token):
        """Get lost of suggested actions for token."""
        # cases when we don't need to do anything
        if prob < self.min_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith('$SUFFIXTRANSFORM_') or sugg_token == '$DELETE':
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith('$SUFFIXTRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob, sugg_token,token

    def _get_embbeder(self, weigths_name, special_tokens_fix):
        embedders = {'bert': PretrainedBertEmbedder(
            pretrained_model=weigths_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=special_tokens_fix)
        }
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=embedders,
            embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
            allow_unmatched_keys=True)
        return text_field_embedder

    def _get_indexer(self, weights_name, special_tokens_fix):
        bert_token_indexer = PretrainedBertIndexer(
            pretrained_model=weights_name,
            do_lowercase=self.lowercase_tokens,
            max_pieces_per_token=5,
            use_starting_offsets=True,
            truncate_long_sequences=True,
            special_tokens_fix=special_tokens_fix,
            is_test=True
        )
        return {'bert': bert_token_indexer}

    def preprocess(self, token_batch):
        seq_lens = [len(sequence) for sequence in token_batch if sequence]
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []
        for indexer in self.indexers:
            batch = []
            for sequence in token_batch:
                tokens = sequence[:max_len]
                tokens = [Token(token) for token in ['$START'] + tokens]
                batch.append(Instance({'tokens': TextField(tokens, indexer)}))
            batch = Batch(batch)
            batch.index_instances(self.vocab)
            batches.append(batch)

        return batches

    def _convert(self, data):
        all_class_probs = torch.zeros_like(data[0]['class_probabilities_labels'])

        error_probs = torch.zeros_like(data[0]['max_error_probability'])
        for output, weight in zip(data, self.model_weights):
            all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)

            error_probs += weight * output['max_error_probability'] / sum(self.model_weights)

        max_vals = torch.max(all_class_probs, dim=-1)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        if self.annotations and  not self.is_ensemble > 1:
            ann_max_vals = torch.max(all_ann_class_probs, dim=-1)
            ann_probs = ann_max_vals[0].tolist()
            ann_idx = ann_max_vals[1].tolist()
        else:
            ann_probs=None
            ann_idx =None


        return probs, idx, error_probs.tolist(), ann_probs, ann_idx

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict, pred_edit, edit_dict, idx):
        new_pred_ids = []
        total_updated = 0
        start_idx = idx + 1 - len(final_batch)
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            edit = pred_edit[i]
            prev_preds = prev_preds_dict[orig_id]
            prev_edits = edit_dict[start_idx + orig_id][1]

            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
                if edit not in prev_edits:
                    edit_dict[start_idx+orig_id][1].extend(edit)
                    if edit_dict[start_idx+orig_id][1][0] == ('$CORRECT'):
                        del edit_dict[start_idx + orig_id][1][0]
            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                if edit not in prev_edits:
                    edit_dict[start_idx+orig_id][1].extend(edit)
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated,edit_dict

    def update_final_ann_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict, pred_edit, edit_dict, ann_error_dict, idx,pred_ann_batch):
        new_pred_ids = []
        total_updated = 0
        start_idx = idx + 1 - len(final_batch)

        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            edit = pred_edit[i]
            ann_pred = pred_ann_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            prev_edits = edit_dict[start_idx + orig_id][1]
            #try:
            prev_ann_preds = ann_error_dict[start_idx + orig_id]
            #except(KeyError):
            #print(start_idx + orig_id)

            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1

                if edit not in prev_edits:
                    edit_dict[start_idx+orig_id][1].extend(edit)
                    if edit_dict[start_idx+orig_id][1][0] == ('$CORRECT'):
                        del edit_dict[start_idx + orig_id][1][0]

                if ann_pred not in prev_ann_preds:
                    new_ann=[]

                    for a,b in zip(ann_pred,prev_ann_preds):
                        if a != '$KEEP':
                            new_ann.append(a)
                        else:
                            new_ann.append(b)

                    ann_error_dict[start_idx+orig_id] = new_ann

            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred

                if edit not in prev_edits:
                    edit_dict[start_idx+orig_id][1].extend(edit)

                if ann_pred not in prev_ann_preds:
                    new_ann = []

                    for a, b in zip(ann_pred, prev_ann_preds):
                        if a != '$KEEP':
                            new_ann.append(a)
                        else:
                            new_ann.append(b)

                    ann_error_dict[start_idx+orig_id] = new_ann
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated, edit_dict, ann_error_dict

    def postprocess_batch(self, batch, all_probabilities, all_idxs,
                          error_probs, encode_verb_form,all_ann_probabilities=None, all_ann_idxs=None):
        all_results = []
        all_edits = []

        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        for tokens, probabilities, idxs, error_prob in zip(batch,
                                                           all_probabilities,
                                                           all_idxs,
                                                           error_probs):
            length = min(len(tokens), self.max_len)
            edits = []

            # skip whole sentences if there no errors
            if max(idxs) == 0:
                all_results.append(tokens)
                all_edits.append([('$CORRECT')])
                continue

            # skip whole sentence if probability of correctness is not high
            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                all_edits.append([('$CORRECT')])
                continue

            for i in range(length + 1):
                # because of START token
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                # skip if there is no error
                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i],
                                                             namespace='labels')
                action = self.get_token_action(token, i, probabilities[i],
                                               sugg_token)
                if not action:
                    continue

                edits.append(action)
            all_edits.append(edits)
            all_results.append(get_target_sent_by_edits(tokens, edits, encode_verb_form))

        return all_results,all_edits

    def handle_batch(self, full_batch, edit_dict, ann_error_dict, idx, encode_verb_form):
        """
        Handle batch of requests.
        """
        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        #prev_preds_edits = {i: [(0,0,'$KEEP',0)] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch))
                     if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        total_updates = 0

        for n_iter in range(self.iterations):
            orig_batch = [final_batch[i] for i in pred_ids]

            sequences = self.preprocess(orig_batch)

            if not sequences:
                break

            probabilities, idxs, error_probs, _, _ = self.predict(sequences)
            pred_batch,edit = self.postprocess_batch(orig_batch, probabilities,
                                                idxs, error_probs, encode_verb_form)
            final_batch, pred_ids, cnt, edit_dict = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict, edit, edit_dict, idx)
            total_updates += cnt
            #edits.append(edit)
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100*len(pred_ids)/batch_size, 1)}% of sentences.")

            if not pred_ids:
                break


        return final_batch, edit_dict, total_updates
