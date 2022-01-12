"""
create .cor file due to the given .ori file and a system output
"""

from lm_scorer import LMScorer
import argparse
import logging

from . import m2, util
from .filepath import Path
import os
logging.basicConfig(level=logging.INFO)


def load_lm(lm_path=Path.lm_path2, lm_databin=Path.lm_databin2):
    args = argparse.Namespace(
        path=lm_path, data=lm_databin,
        fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0,
        fp16_scale_window=None, fpath=None, future_target=False,
        gen_subset='test', lazy_load=False, log_format=None, log_interval=1000,
        max_sentences=None, max_tokens=None, memory_efficient_fp16=False,
        min_loss_scale=0.0001, model_overrides='{}', no_progress_bar=True,
        num_shards=1, num_workers=0, output_dictionary_size=-1,
        output_sent=False, past_target=False,
        quiet=True, raw_text=False, remove_bpe=None, sample_break_mode=None,
        seed=1, self_target=False, shard_id=0, skip_invalid_size_inputs_valid_test=False,
        task='language_modeling', tensorboard_logdir='', threshold_loss_scale=None,
        tokens_per_sample=1024, user_dir=None, cpu=False)
    return LMScorer(args)


def postprocess(ori_path, system_out, cor_path, lm_path, lm_data, remove_unk_edits=True, remove_error_type_lst=[],
                apply_rerank=False, preserve_spell=False, max_edits=None,infer_boosting=False):
    #system_out = util.get_basename(system_out, include_path=True, include_extension=False)
    #system_out_short = system_out.replace('_bpe60-','6').replace('_bpe50-','5').replace('_bpe60_','6').replace('_bpe50_','5').replace('_','')
    system_out_short = util.shorten_name(system_out)
    pred = f"{system_out_short}.pred"
    m2_file_tmp = f"{system_out_short}._m2"

    logging.info("[Postprocess] 1. get pred file")
    m2.sys_to_cor(system_out, pred,infer_boosting)

    logging.info("[Postprocess] 2. convert pred into m2")
    m2.parallel_to_m2(ori_path, pred, m2_file_tmp)

    logging.info("[Postprocess] 3. adjust m2")
    m2_entries = m2.get_m2_entries(m2_file_tmp)

    if remove_unk_edits:
        logging.info("[Postprocess] 3-1. removing <unk> edits")
        m2_entries = m2.remove_m2(m2_entries, None, '<unk>')
    print(f" the errors to be removed: {remove_error_type_lst}")
    if len(remove_error_type_lst) > 0:
        logging.info("[Postprocess] 3-2. remove error types")
        m2_entries = m2.remove_m2(m2_entries, remove_error_type_lst, None)

    if apply_rerank:
        logging.info("[Postprocess] 3-3. apply rerank")
        lm_scorer = load_lm(lm_path,lm_data)
        m2_entries = m2.apply_lm_rerank(m2_entries, preserve_spell, max_edits, lm_scorer)

    logging.info("[Postprocess] 4. get pred again")
    logging.info("[Postprocess] 4-1. write m2 file")

    cor_basename = util.get_basename(cor_path, include_extension=False)
    m2_file = f"{cor_basename}.m2"
    m2.write_m2_entries(m2_entries, m2_file)

    logging.info("[Postprocess] 4-2. write cor file")
    m2.m2_to_parallel([m2_file], None, cor_path, False, True)
    if apply_rerank:
        os.system(f" zip -j {cor_path}.zip {cor_path}")

