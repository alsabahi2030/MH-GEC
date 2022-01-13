import argparse
from gec.filepath import Path
from gec import util, m2
from gector.gec_model_ann import GecBERTModel
import logging
import os
import json
from tqdm import tqdm
import time
from utils.helpers import read_lines, get_verb_form_dicts

logging.basicConfig(level=logging.INFO)
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
def predict_for_file(input_file, output_file, edits_file,annotation_file, model, verb_form_dict, batch_size=32):
    test_data = read_lines(input_file)
    if args.preprocess:
        test_data = [preprocessing(t) for t in test_data]

    predictions = []
    edit_dict ={}
    ann_error_dict = {}
    edits =[]
    cnt_corrections = 0
    batch = []
    for idx, sent in enumerate(test_data):
        batch.append(sent.split())
        edit_dict[idx]= sent, [('$CORRECT')]
        if len(batch) == batch_size:
            preds,edit_dict, cnt = model.handle_batch(batch,edit_dict,ann_error_dict,idx,verb_form_dict)

            predictions.extend(preds)

            cnt_corrections += cnt
            batch = []
    if batch:
        preds, edit_dict, cnt = model.handle_batch(batch, edit_dict,ann_error_dict, idx,verb_form_dict)
        predictions.extend(preds)
        #edits.extend(edit)
        cnt_corrections += cnt
    predictions = "\n".join([" ".join(x) for x in predictions])
    with open(output_file, 'w', encoding='utf8') as f:
        f.write(predictions)
    with open(edits_file, 'w') as f1:
        json.dump(edit_dict, f1)

    return cnt_corrections

def main(args):
    # get all paths
    output_file = None
    if args.subset == 'valid':
        gold_m2_file = "./data/m2/ABCN.dev.gold.bea19.m2"
        # gold_m2_file = "./eval_data/ABCN.dev.gold.bea19.m2"

        # input_file ="./data/wi.dev.ori"
        input_file = "./data/wi.dev.ori"
        # input_file ="./eval_data/wi.dev.ori"
        scorer_type = 'errant'
    elif args.subset == 'coll2013':
        gold_m2_file = "./new_data/coll2013.gold.m2"
        input_file = "./data/raw/conll2013.ori"
        scorer_type = 'm2scorer'

    elif args.subset == 'conll2014':
        gold_m2_file = "./data/m2/official-2014.combined.m2"
        input_file = "./data/raw/conll2014.ori"
        scorer_type = 'm2scorer'

    elif args.subset == 'testorg':
        input_file = "./data/ABCN.bea19.test.orig.txt"
        input_file = "./data/raw/ABCN.test.bea19.orig"
        scorer_type = None
    else:
        input_file = args.input_file

    if not args.find_best:
        start_time = time.time()
        verb_form_dict = get_verb_form_dicts('./resources/verb-form-vocab.txt')
        model_paths = []
        if args.model_dir is not None:
            if not args.is_ensemble:
                #ckpt_dirs = [k for k in ckpt_dirs if k.startswith('model_state_epoch')]
                ckpt_dirs = util.get_sorted_ckpts(args.model_dir, args.epoch_start, args.epoch_end, args.epoch_interval)
            else:
                ckpt_dirs = os.listdir(args.model_dir)
                ckpt_dirs = [k for k in ckpt_dirs if os.path.isfile(os.path.join(args.model_dir, k))]
                if args.transformer_model is not None:
                    ckpt_dirs = [k for k in ckpt_dirs if args.transformer_model in k]
                if args.stage:
                    ckpt_dirs = [k for k in ckpt_dirs if args.stage in k]

            for ckpt in ckpt_dirs:
                model_paths.append([os.path.join(args.model_dir,ckpt)])
        else:
            model_paths.append(args.model_path)
        if args.output_dir is None:
            if args.is_ensemble:
                output_dir = f"./outputs/outputs_ensemble_{args.is_ensemble}_{args.subset}_errant{args.errant}"
                os.makedirs(output_dir, exist_ok=True)

                if args.multiclassifier2:
                    output_dir += '/mc2'
                elif args.multiclassifier:
                    output_dir += '/mc1'
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = f"./outputs"
        else:
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)

        #os.makedirs(output_dir, exist_ok=True)
        cor_files =[]
        report_files =[]
        for model_path in tqdm(model_paths):
            #try:
            logging.info(f'Model path:{model_path}')
            if args.output_file is None:
                output_file = None
                data_basename = util.get_basename(input_file, include_path=False, include_extension=False)
                if not args.is_ensemble:
                    output_dir = f"./outputs"
                    ckpt_basename = util.get_basename(''.join(model_path), include_path=False, include_extension=False)
                    model_path_base = os.path.dirname(model_path[0])
                    model_path_base = model_path_base.split('/')[-1]
                    desc = model_path_base.split("_")
                    if 'xlnet' in desc[1]:
                        special_tokens_fix = 0
                    else:
                        special_tokens_fix = 1
                    transformer_model = desc[1]
                    round=desc[4].split('/')[0]
                    if args.multiclassifier2:
                        output_dir += '/mc2'
                    elif args.multiclassifier:
                        output_dir += '/mc1'
                    elif args.early_exit2:
                        output_dir += '/ex2'
                    elif args.early_exit1:
                        output_dir += '/ex1'

                    output_dir = f"{output_dir}/{desc[3]}_{args.subset}_errant{args.errant}"
                    os.makedirs(output_dir, exist_ok=True)
                    if output_file is None:
                        output_file = os.path.join(output_dir, f"{desc[1]}.{desc[2]}.{desc[3]}.{round}.{ckpt_basename}.{data_basename}")
                    #edits_file = os.path.join(output_dir, f"{desc[1]}.{desc[2]}.{desc[3]}.{round}.{ckpt_basename}.{data_basename}.edit")
                    #annotation_file = os.path.join(output_dir, f"{desc[1]}.{desc[2]}.{desc[3]}.{round}.{ckpt_basename}.{data_basename}.ann")

                else:
                    transformer_model=args.transformer_model
                    special_tokens_fix=args.special_tokens_fix
                    ckpt_name_list=[]
                    for m in model_path:
                        ckpt_basename = util.get_basename(''.join(m), include_path=False, include_extension=False)
                        desc = ckpt_basename.split("_")
                        if args.is_ensemble < 5:
                            ckpt_name_list.append(f"{desc[0]}.{desc[2]}.{desc[3]}.ep{desc[5]}")
                        else:
                            ckpt_name_list.append(f"{desc[0][0]}.{desc[2][8:]}.{desc[3]}.e{desc[5]}")

                    output_file_name = "_".join(ckpt_name_list)
                    if len(output_file_name) > 200:
                        ckpt_name_list = []
                        for m in model_path:
                            ckpt_basename = util.get_basename(''.join(m), include_path=False, include_extension=False)
                            desc = ckpt_basename.split("_")
                            if '-' in desc[0]:
                                t =f"{desc[0][0]}l"
                            else:
                                t=desc[0][0]
                            ckpt_name_list.append(f"{t}.{desc[2][8:]}.{desc[3]}.e{desc[5]}")
                        output_file_name = "_".join(ckpt_name_list)
                    if output_file is None:
                        output_file = os.path.join(output_dir, f"{output_file_name[0:200]}.{data_basename}")
                    edits_file = os.path.join(output_dir, f"{output_file_name[0:200]}.{data_basename}")
                    annotation_file = os.path.join(output_dir, f"{output_file_name[0:200]}.{data_basename}")
            else:
                output_file = os.path.join(output_dir,args.output_file)
                transformer_model = args.transformer_model
                special_tokens_fix = args.special_tokens_fix
            vocab_path = args.vocab_path
            if args.max_len == 50:
                cor_path = f"{output_file}.mp{args.min_error_probability}.cf{args.additional_confidence}.cor"
                edits_file = f"{output_file}.mp{args.min_error_probability}.cf{args.additional_confidence}.gedit"

            else:
                if args.weights:
                    w = [str(t) for t in args.weights]
                    w = '.'.join(w)
                    cor_path = f"{output_file}.mp{args.min_error_probability}.cf{args.additional_confidence}x{args.max_len}.{w}.cor"
                    edits_file = f"{output_file}.mp{args.min_error_probability}.cf{args.additional_confidence}x{args.max_len}.{w}.gedit"
                elif args.lamda < 1:
                    cor_path = f"{output_file}.mp{args.min_error_probability}.cf{args.additional_confidence}x{args.max_len}.l{args.lamda}.cor"
                    edits_file = f"{output_file}.mp{args.min_error_probability}.cf{args.additional_confidence}x{args.max_len}.l{args.lamda}.gedit"
                else:
                    cor_path = f"{output_file}.mp{args.min_error_probability}.cf{args.additional_confidence}x{args.max_len}.cor"
                    edits_file = f"{output_file}.mp{args.min_error_probability}.cf{args.additional_confidence}x{args.max_len}.gedit"
                cor_basename = util.get_basename(cor_path, include_extension=False)
            cor_files.append(cor_path)
            if (not os.path.isfile(cor_path) and not args.find_best) or args.overwrite:
                model = GecBERTModel(vocab_path=vocab_path,
                                 model_paths=model_path,
                                 max_len=args.max_len, min_len=args.min_len,
                                 iterations=args.iteration_count,
                                 min_error_probability=args.min_error_probability,
                                 min_probability=args.min_error_probability,
                                 lowercase_tokens=args.lowercase_tokens,
                                 model_name=transformer_model,
                                 special_tokens_fix=special_tokens_fix,
                                 log=False,
                                 confidence=args.additional_confidence,
                                 twolaysers_classifier=args.twolaysers_classifier,
                                 multiclassifier=args.multiclassifier,
                                 multiclassifier2=args.multiclassifier2,
                                 early_exit1=args.early_exit1,
                                 early_exit2=args.early_exit2,
                                 multiheadversion=args.multiheadversion,
                                 lamda=args.lamda,

                                 relu_activation=args.relu_activation,
                                 is_ensemble=args.is_ensemble,
                                 weigths=args.weights,
                                 cfs=args.cfs, predictonly=args.predictonly,use_cpu=bool(args.use_cpu))

                cnt_corrections = predict_for_file(input_file, cor_path,edits_file,annotation_file, model,verb_form_dict,
                                                   batch_size=args.batch_size)
                # evaluate with m2 or ERRANT

                print(f"Produced overall corrections: {cnt_corrections}")
                cor_basename = util.get_basename(cor_path, include_extension=False)
                if scorer_type == 'errant':
                    report_path = f"{cor_basename}.report"
                elif scorer_type == 'm2scorer':
                    report_path = f"{cor_basename}.m2report"
                report_files.append(report_path)
            end_time = time.time()
            logging.info(f"prediction time： {end_time - start_time}")
            if len(args.remove_error_type_lst) > 0:
                m2_file_tmp = f"{cor_path}._m2"
                logging.info("[Postprocess] 2. convert pred into m2")
                m2.parallel_to_m2(input_file, cor_path, m2_file_tmp)

                logging.info("[Postprocess] 3. adjust m2")
                m2_entries = m2.get_m2_entries(m2_file_tmp)
                logging.info("[Postprocess] 3-2. remove error types")
                m2_entries = m2.remove_m2(m2_entries, args.remove_error_type_lst, None)
                cor_basename = util.get_basename(cor_path, include_extension=False)
                m2_file = f"{cor_basename}.m2"
                m2.write_m2_entries(m2_entries, m2_file)

                logging.info("[Postprocess] 4-2. write cor file")
                m2.m2_to_parallel([m2_file], None, cor_path, False, True)
        else:
            logging.info('Output directory can\'t be empty')

if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+')
    parser.add_argument('--vocab_path',
                        help='Path to the model file.' # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file')
    parser.add_argument('--output_file',
                        help='Path to the output file')
    parser.add_argument('--model-dir',
                        help='Path to the model dirs')
    parser.add_argument('--output-dir',
                        help='Path to the output file')
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'distilbert', 'gpt2', 'roberta', 'transformerxl', 'xlnet', 'albert', 'roberta-large', 'xlnet-large', 'deberta', 'deberta-large', 'bart', 'bart-large', 'bert-large', 't5-base', 'funnel-transformer-medium-base', 'roberta-openai', 'deberta-xx-large', 'deberta-xlarge', 'ukr-roberta-base'],
                        help='Name of the transformer model.')
    parser.add_argument('--training_mode',
                        help='training stages')
    parser.add_argument('--pretrain_data',
                        help='dataset')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--min_probability',
                        type=float,
                        default=0.0)
    parser.add_argument('--cfs',  nargs='+',help ='a list of additional_confidence for Roberta and Xlnet')

    parser.add_argument('--lamda',
                        type=float,
                        default=1.0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--multiheadversion',
                        type=int,
                        help='The order of the multihead classifier in early exiting.',
                        default=1, choices=[1,2,3,4])
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--errant',
                        type=int,
                        help='errant version.',
                        default=2)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument("--remove-error-type-lst", type=str, nargs="+", default=[],
                        help="error types to be removed (e.g.. R:OTHER)")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--stage", type=str)
    parser.add_argument("--notin", type=str)

    parser.add_argument("--find-best", action="store_true")
    parser.add_argument("--twolaysers-classifier", action="store_true")
    parser.add_argument("--relu_activation", action="store_true")

    parser.add_argument("--basic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--explanatory", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--multiclassifier", action="store_true")
    parser.add_argument("--multiclassifier2", action="store_true")
    parser.add_argument("--early_exit1", action="store_true")
    parser.add_argument("--early_exit2", action="store_true")
    parser.add_argument("--evaluate_all", action="store_true")
    parser.add_argument("--predictonly", action="store_true")

    parser.add_argument("--copy-testing", action="store_true")
    parser.add_argument("--for_combining", action="store_true")
    parser.add_argument("--do_testing", action="store_true")
    parser.add_argument("--gpuno", type=int, default=0)

    parser.add_argument("--annotations", action="store_true")
    parser.add_argument("--copy-eval_path", help='Path to the copied files')
    parser.add_argument("--copy-best_ckpts", action="store_true", help='whether to copy best ckpts')
    parser.add_argument("--epoch_interval", type=int, default=1, help="")

    parser.add_argument("--epoch_start", type=int, default=0, help="")
    parser.add_argument("--epoch_end", type=int, default=999999999, help="")
    parser.add_argument('--use_cpu',
                        type=int,
                        help='use only cpu',
                        default=0)
    parser.add_argument("--number-of-ckpt", type=int, default=3, help="")
    args = parser.parse_args()
    main(args)
