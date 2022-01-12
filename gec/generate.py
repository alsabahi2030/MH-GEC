import os


def generate(data_path, ckpt, system_out,infer_boosting=False,nbest=1, ori_path=None, gen_subset=None, beam=12, max_tokens=6000, buffer_size=6000 ,is_cpu=False,sup_attn=False):
    """
    :param data_path: data-bin path
    :param ckpt: checkpoint path
    :param system_out: system out path to be created
    :param ori_path: bpe-tokenized ori file path (for fairseq-interactive)
    :param gen_subset: subset of the data-bin path (for fairseq-generate)
    :param beam: beam size
    :param max_tokens: max tokens
    :param buffer_size: buffer size
    :return:
    """

    if ori_path is not None:
        generate = f"fairseq-interactive {data_path} --path {ckpt} --input {ori_path} " \
                   f"--beam {beam} --nbest {nbest} --max-tokens {max_tokens} --buffer-size {buffer_size} > {system_out}"
        if is_cpu:
            generate = f"fairseq-interactive {data_path} --path {ckpt} --input {ori_path} " \
                       f"--beam {beam} --nbest {nbest} --max-tokens {max_tokens} --buffer-size {buffer_size} --cpu > {system_out}"
        os.system(generate)

    elif gen_subset is not None:
        if infer_boosting:
            generate = f"python fairseq/generate_inference_boosting_v4.py {data_path} --path {ckpt} --gen-subset {gen_subset} " \
                       f"--beam {beam} --nbest 3 --max-tokens {max_tokens} --print-alignment > {system_out}"

        elif sup_attn:
            generate = f"python fairseq/generate_or_copy_v2.py {data_path} --path {ckpt} --gen-subset {gen_subset} " \
                       f"--beam {beam} --nbest {nbest} --max-tokens {max_tokens} --print-alignment > {system_out}"
        else:
            generate = f"fairseq-generate {data_path} --path {ckpt} --gen-subset {gen_subset} " \
                       f"--beam {beam} --nbest {nbest} --max-tokens {max_tokens} --print-alignment > {system_out}"
        if is_cpu:
            if infer_boosting:
                generate = f"python fairseq/generate_inference_boosting_v4.py  {data_path} --path {ckpt} --gen-subset {gen_subset} " \
                           f"--beam {beam} --nbest 3 --max-tokens {max_tokens} --print-alignment --cpu > {system_out}"
            else:
                generate = f"fairseq-generate {data_path} --path {ckpt} --gen-subset {gen_subset} " \
                           f"--beam {beam} --nbest {nbest} --max-tokens {max_tokens} --print-alignment --cpu > {system_out}"
        os.system(generate)
