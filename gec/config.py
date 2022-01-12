def get_model_config(model, lr, dropout, max_epoch, seed, reset=False):
    assert model in ['base', 'copy', 't2t', 'copy_el', 'copy_el_att', 'copy_big', 'base_small']
    if model == 'base':
        model_config = f"--arch transformer --share-all-embeddings " \
                       f"--optimizer adam --lr {lr} --label-smoothing 0.1 --dropout {dropout} " \
                       f"--max-tokens 3500 --min-lr '1e-09' --lr-scheduler inverse_sqrt " \
                       f"--weight-decay 0.0001 --criterion label_smoothed_cross_entropy " \
                       f"--max-epoch {max_epoch} --warmup-updates 4000 --warmup-init-lr '1e-07' " \
                       f"--encoder-layers 6 --encoder-embed-dim 1024 --decoder-layers 6 --decoder-embed-dim 1024 " \
                       f"--adam-betas '(0.9, 0.98)' --save-interval-updates 5000 --update-freq 7 "

    elif model == 'base_small':
        model_config = f"--arch transformer --share-all-embeddings " \
                       f"--optimizer adam --lr {lr} --label-smoothing 0.1 --dropout {dropout} " \
                       f"--max-tokens 3500 --min-lr '1e-09' --lr-scheduler inverse_sqrt " \
                       f"--weight-decay 0.0001 --criterion label_smoothed_cross_entropy " \
                       f"--max-epoch {max_epoch} --warmup-updates 4000 --warmup-init-lr '1e-07' " \
                       f"--adam-betas '(0.9, 0.98)' --save-interval-updates 5000 --update-freq 7 "

    elif model == 'copy':
        model_config = f"--ddp-backend=no_c10d --arch copy_augmented_transformer " \
                       f"--update-freq 12 --alpha-warmup 10000 --optimizer adam --lr {lr} " \
                       f"--dropout {dropout} --max-tokens 3000 --min-lr '1e-09' --save-interval-updates 5000 " \
                       f"--lr-scheduler inverse_sqrt --weight-decay 0.0001 --max-epoch {max_epoch} " \
                       f"--warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' "

    elif model == 'copy_big':
        model_config = f"--ddp-backend=no_c10d --arch copy_augmented_transformer " \
                       f"--update-freq 12 --alpha-warmup 10000 --optimizer adam --lr {lr} " \
                       f"--dropout {dropout} --max-tokens 2000 --min-lr '1e-09' --save-interval-updates 5000 " \
                       f"--lr-scheduler inverse_sqrt --weight-decay 0.0001 --max-epoch {max_epoch} " \
                       f"--encoder-layers 6 --encoder-embed-dim 1024 --decoder-layers 6 --decoder-embed-dim 1024 " \
                       f"--warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' "
    elif model == 'copy_el':
        model_config = f"--ddp-backend=no_c10d " \
                       f"--arch copy_augmented_transformer_aux_el --task gec_labels --criterion gec_loss " \
                       f"--edit-weighted-loss 1.0 --edit-label-prediction 1.0 " \
                       f"--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 " \
                       f"--lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --max-epoch {max_epoch} " \
                       f"--warmup-updates 4000 --lr {lr} --min-lr '1e-09' --dropout {dropout} " \
                       f"--weight-decay 0.0 --max-tokens 2000 --save-interval-updates 5000 --update-freq 8 "

    elif model == 'copy_el_att':
        model_config = f"--ddp-backend=no_c10d " \
                       f"--arch copy_augmented_transformer_aux_el_supervisedAtt --task gec_labels --criterion gec_loss " \
                       f"--edit-weighted-loss 1.0 --edit-label-prediction 1.0 " \
                       f"--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 " \
                       f"--lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --max-epoch {max_epoch} " \
                       f"--warmup-updates 4000 --lr {lr} --min-lr '1e-09' --dropout {dropout} " \
                       f"--weight-decay 0.0 --max-tokens 2000 --save-interval-updates 5000 --update-freq 8 "

    else:  # model == 't2t':
        model_config = f"--arch transformer_wmt_en_de_big_t2t --share-all-embeddings " \
                       f"--criterion label_smoothed_cross_entropy --label-smoothing 0.1 " \
                       f"--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 " \
                       f"--lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --max-epoch {max_epoch} " \
                       f"--warmup-updates 4000 --lr {lr} --min-lr '1e-09' --dropout {dropout} " \
                       f"--weight-decay 0.0 --max-tokens 2000 --save-interval-updates 3000 --update-freq 12 "

    if seed is not None:
        model_config += f"--seed {seed} "
    if reset:
        model_config += f"--reset-optimizer --reset-lr-scheduler "

    return model_config

