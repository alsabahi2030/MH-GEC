from abc import abstractmethod
import os

from . import util
from .filepath import FilePath


def choice_track(track_num):
    if track_num == 0:
        return Track0()

    if track_num == 1:
        return Track1()

    if track_num == 4:
        return Track4()
    if track_num == 5:
        return Track5()
    
    if track_num == 3:
        return Track3()


class Track:
    def __init__(self, track_num):
        self.fp = FilePath()
        self.TRACK_NUM = track_num
        self.TRACK_PATH = f"{self.fp.root}/track{track_num}"

    @property
    def train_modes(self):
        raise NotImplementedError

    @property
    def subsets(self):
        raise NotImplementedError

    def get_databin_path(self, train_mode, bpe_version):
        assert train_mode in self.train_modes
        if bpe_version == 32:
            return f"{self.TRACK_PATH}/data-bin32/{train_mode}"
        else:
            return f"{self.TRACK_PATH}/data-bin{bpe_version}/{train_mode}"


    def get_ckpt_dir(self, train_mode, model, lr=5e-4, dropout=0.3, seed=None, prev_model_dir=None,bpe_version=50,edit_label_factor=1.0,update_freq=1):

        def _get_ckpt_dir_basename(train_mode, model, lr, dropout, seed, prev_model_dir,bpe_version,update_freq):
            basenames = []
            if prev_model_dir is not None:
                prev_model_basename = util.get_basename(prev_model_dir, include_path=False, include_extension=False)
                basenames.append(prev_model_basename)

            basename = f"{train_mode}-{model}-bpe{bpe_version}-lr{lr}-dr{dropout}-upd{update_freq}"
            if seed is not None:
                basename += f"-s{seed}"
            if 'att' in  model:
                basename += f"-el{edit_label_factor}"
            basenames.append(basename)

            return "_".join(basenames)

        ckpt_basename = _get_ckpt_dir_basename(train_mode, model, lr, dropout, seed, prev_model_dir, bpe_version,update_freq)

        return f"{self.TRACK_PATH}/ckpt{bpe_version}/{ckpt_basename}"

    def get_output_dir(self, ckpt,bpe_version):
        def _get_output_dir_from_ckpt_dir(ckpt_dir):
            dir_basename = util.get_basename(ckpt_dir, include_path=False)
            return f"{self.TRACK_PATH}/outputs{bpe_version}/{dir_basename}"

        def _get_output_dir_from_ckpt_fpath(ckpt_fpath):
            ckpts = ckpt_fpath.split(':')

            # not ensemble
            if len(ckpts) == 1:
                ckpt_dir = os.path.dirname(ckpt_fpath)
                return _get_output_dir_from_ckpt_dir(ckpt_dir)

            # ensemble
            else:
                dirname_lst = []
                for ckpt in ckpts:
                    ckpt_dir = os.path.dirname(ckpt)
                    ckpt_dir_basename = util.get_basename(ckpt_dir, include_path=False)
                    dirname_lst.append(ckpt_dir_basename)
                return f"{self.TRACK_PATH}/outputs/" + ":".join(dirname_lst)

        if os.path.isdir(ckpt):
            return _get_output_dir_from_ckpt_dir(ckpt)
        else:
            return _get_output_dir_from_ckpt_fpath(ckpt)

    @abstractmethod
    def get_subset_datapath(self, subset):
        raise NotImplementedError

    @staticmethod
    def get_model_config(model, lr, dropout, max_epoch, seed, reset=False,edit_label_factor=1.0,fp16 = False, update_freq=1):
        assert model in ['base', 'copy', 't2t','copy_el','copy_el_att','copy_big','base_small','copy_el_att2','base4_lh','copy_4lh','copy_4lh2','copy_el_att2_4lh']
        if model == 'base':
            model_config = f"--arch transformer --share-all-embeddings " \
                 f"--optimizer adam --lr {lr} --label-smoothing 0.1 --dropout {dropout} " \
                 f"--max-tokens 3500 --min-lr '1e-09' --lr-scheduler inverse_sqrt " \
                 f"--weight-decay 0.0001 --criterion label_smoothed_cross_entropy " \
                 f"--max-epoch {max_epoch} --warmup-updates 4000 --warmup-init-lr '1e-07' " \
                 f"--encoder-layers 6 --encoder-embed-dim 1024 --decoder-layers 6 --decoder-embed-dim 1024 " \
                 f"--adam-betas '(0.9, 0.98)' --save-interval-updates 5000 --update-freq  {update_freq} "

        elif model == 'base4_lh':
            model_config = f"--arch transformer_4_layers_heads --share-all-embeddings " \
                 f"--optimizer adam --lr {lr} --label-smoothing 0.1 --dropout {dropout} " \
                 f"--max-tokens 5820 --min-lr '1e-09' --lr-scheduler inverse_sqrt " \
                 f"--weight-decay 0.0001 --criterion label_smoothed_cross_entropy " \
                 f"--max-epoch {max_epoch} --warmup-updates 4000 --warmup-init-lr '1e-07' " \
                 f"--adam-betas '(0.9, 0.98)' --save-interval-updates 5000 --update-freq {update_freq} "

        elif model == 'base_small':
            model_config = f"--arch transformer --share-all-embeddings " \
                 f"--optimizer adam --lr {lr} --label-smoothing 0.1 --dropout {dropout} " \
                 f"--max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt " \
                 f"--weight-decay 0.0001 --criterion label_smoothed_cross_entropy " \
                 f"--max-epoch {max_epoch} --warmup-updates 4000 --warmup-init-lr '1e-07' " \
                 f"--adam-betas '(0.9, 0.98)' --save-interval-updates 5000 --update-freq  {update_freq} "

        elif model == 'copy_4lh':
            model_config = f"--ddp-backend=no_c10d --arch copy_augmented_transformer_4layers_heads " \
                f"--update-freq {update_freq} --alpha-warmup 10000 --optimizer adam --lr {lr} --label-smoothing 0.1 " \
                f"--dropout {dropout} --max-tokens 4000 --min-lr '1e-09' --save-interval-updates 5000 " \
                f"--lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-epoch {max_epoch} " \
                f"--warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' "


        elif model == 'copy_4lh2':
            model_config = f"--ddp-backend=no_c10d --arch copy_augmented_transformer_4layers_heads " \
                f"--update-freq {update_freq} --alpha-warmup 10000 --optimizer adam --lr {lr}  " \
                f"--dropout {dropout} --max-tokens 4000 --min-lr '1e-09' --save-interval-updates 5000 " \
                f"--lr-scheduler inverse_sqrt --weight-decay 0.0001  --max-epoch {max_epoch} " \
                f"--warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' "


        elif model == 'copy':
            model_config = f"--ddp-backend=no_c10d --arch copy_augmented_transformer " \
                f"--update-freq {update_freq} --alpha-warmup 10000 --optimizer adam --lr {lr} " \
                f"--dropout {dropout} --max-tokens 2000 --min-lr '1e-09' --save-interval-updates 5000 " \
                f"--lr-scheduler inverse_sqrt --weight-decay 0.0001 --max-epoch {max_epoch} " \
                f"--warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' "

        elif model == 'copy_big':
            model_config = f"--ddp-backend=no_c10d --arch copy_augmented_transformer " \
                f"--update-freq {update_freq} --alpha-warmup 10000 --optimizer adam --lr {lr} " \
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
               f"--weight-decay 0.0 --max-tokens 2000 --save-interval-updates 5000 --update-freq  {update_freq} "

        elif model == 'copy_el_att':
            model_config = f"--ddp-backend=no_c10d " \
               f"--arch copy_augmented_transformer_aux_el_supervisedAtt --task gec_labels --criterion gec_loss " \
               f"--edit-weighted-loss 1.0 --edit-label-prediction {edit_label_factor} " \
               f"--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 " \
               f"--lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --max-epoch {max_epoch} " \
               f"--warmup-updates 4000 --lr {lr} --min-lr '1e-09' --dropout {dropout} " \
               f"--weight-decay 0.0 --max-tokens 3000 --save-interval-updates 5000  --update-freq 8 "

        elif model == 'copy_el_att2':
            model_config = f"--ddp-backend=no_c10d " \
               f"--arch copy_augmented_transformer_aux_el_supervisedAtt2 --task gec_labels --criterion gec_loss_copy_attention " \
               f"--edit-weighted-loss 1.0 --edit-label-prediction {edit_label_factor} " \
               f"--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 " \
               f"--lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --max-epoch {max_epoch} " \
               f"--warmup-updates 4000 --lr {lr} --min-lr '1e-09' --dropout {dropout} " \
               f"--weight-decay 0.0 --max-tokens 3000 --save-interval-updates 20000  --update-freq {update_freq} "

        elif model == 'copy_el_att2_4lh':
            model_config = f"--ddp-backend=no_c10d " \
               f"--arch copy_augmented_transformer_aux_el_supervisedAtt2_4lh --task gec_labels --criterion gec_loss_copy_attention " \
               f"--edit-weighted-loss 1.0 --edit-label-prediction {edit_label_factor} " \
               f"--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 " \
               f"--lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --max-epoch {max_epoch} " \
               f"--warmup-updates 4000 --lr {lr} --min-lr '1e-09' --dropout {dropout} " \
               f"--weight-decay 0.0 --max-tokens 4000 --save-interval-updates 5000  --update-freq {update_freq}  "

        else:   # model == 't2t':
            model_config = f"--arch transformer_wmt_en_de_big_t2t --share-all-embeddings " \
                           f"--criterion label_smoothed_cross_entropy --label-smoothing 0.1 " \
                           f"--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 " \
                           f"--lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --max-epoch {max_epoch} " \
                           f"--warmup-updates 4000 --lr {lr} --min-lr '1e-09' --dropout {dropout} " \
                           f"--weight-decay 0.0 --max-tokens 2000 --save-interval-updates 3000 --update-freq 16 "

        if seed is not None:
            model_config += f"--seed {seed} "
        if reset:
            model_config += f"--reset-optimizer --reset-lr-scheduler "
        if fp16:
            model_config += f"--fp16  "

        return model_config


class Track0(Track):
    def __init__(self):
        super(Track0, self).__init__(0)

    train_modes = ['pretrain', 'train', 'finetune']
    subsets = ['valid', 'conll2014', 'jfleg','train']

    def get_pref(self, train_mode):
        assert train_mode in self.train_modes
        if train_mode == 'pretrain':
            trainpref = os.path.splitext(self.fp.DAE_ORI0)[0]
        elif train_mode == 'train':
            trainpref = os.path.splitext(self.fp.TRAIN_ORI0)[0]
        else:  # finetune
            trainpref = os.path.splitext(self.fp.FINETUNE_ORI0)[0]
        validpref = os.path.splitext(self.fp.VALID_ORI0)[0]
        return trainpref, validpref

    def get_subset_datapath(self, subset):
        assert subset in self.subsets

        if subset == 'valid':
            gold_m2 = f"{self.fp.conll2013_m2}/official-preprocessed.m2"
            ori_path = self.fp.CONLL2013_ORI
            ori_bpe_path = None
            gen_subset = "valid"
            scorer_type = "m2scorer"

        if subset == 'train':
            gold_m2 = f"{self.fp.wi_m2}/ABCN.train.gold.bea19.m2"
            ori_path = self.fp.WI_TRAIN_ORI
            ori_bpe_path = None
            gen_subset = "train"
            scorer_type = "m2scorer"

        elif subset == 'conll2014':
            gold_m2 = f"{self.fp.conll2014_m2}/official-2014.combined.m2"
            ori_path = self.fp.CONLL2014_ORI
            ori_bpe_path = self.fp.CONLL2014_TOK_ORI
            gen_subset = None
            scorer_type = "m2scorer"

        else:  # 'jfleg':
            gold_m2 = None
            ori_path = self.fp.JFLEG_ORI
            ori_bpe_path = self.fp.JFLEG_TOK_ORI
            gen_subset = None
            scorer_type = "jfleg"

        return gold_m2, ori_path, ori_bpe_path, gen_subset, scorer_type


class Track1(Track):
    def __init__(self):
        super(Track1, self).__init__(1)

    #train_modes = ['pretrain', 'train', 'trainbig', 'finetune', 'dev','learnfrommistakes', 'train_learnfrommistakes','learnfrommistakes_copy','train_learnfrommistakes_2nd142','train_learnfrommistakes_bs204','cvtefinetune','cvtebigfinetune','cvtetrain','cvtefinetune2', 'lfm_1st-copy-small-lfm-bpe50-ckpt_best','lfm_2nd_lfm142_t2t_bpe50_ckpt148','lfm_1st_base-big_bpe50_ckpt92','lfm_1st_copy_el_att_ckpt107','lfm_1st_base_small_ckpt29','lfm_1st_base_small_ckpt29_2' ,']
    train_modes = ['pretrain', 'train', 'trainbig', 'finetune','lfm_1st_copy_bpe60_ckpt174','lfm_1st_base_samll_bpe60_ckpt187','lfm_2nd_t2t_bpe60_ckpt220','lfm_2nd_copy_4lh_ckpt161','lfm_2ndCopy_ckpt176']

    subsets = ['valid', 'test','trainbig', 'conll2014','train','cvte_m2','lwwtest','cvte_errant','valid3481','onlinetest','testorg','valid2nd','devorg','soncan']

    def get_pref(self, train_mode,bpe_version=50 ):
        assert train_mode in self.train_modes
        if bpe_version == 32:
            if train_mode == 'pretrain':
                trainpref = os.path.splitext(self.fp.DAE32_ORI1)[0]
            elif train_mode == 'train':
                trainpref = os.path.splitext(self.fp.TRAIN32_ORI1)[0]
            elif train_mode == 'finetune':  # finetune
                trainpref = os.path.splitext(self.fp.FINETUNE32_ORI1)[0]
            elif train_mode == 'learnfrommistakes':
                trainpref = os.path.splitext(self.fp.TRAIN_LFM32_ORI1)[0]
            elif train_mode == 'train_learnfrommistakes':
                trainpref = os.path.splitext(self.fp.TRAIN_LFM32_ORI1)[0]
                testpref = os.path.splitext(self.fp.TEST32_ORI1)[0]
            elif train_mode == 'cvtefinetune':
                trainpref = os.path.splitext(self.fp.CVTE32_ORI1)[0]
                #testpref = os.path.splitext(self.fp.TEST32_ORI1)[0]

            else:
                trainpref = os.path.splitext(self.fp.VALID32_ORI1)[0]
            validpref = os.path.splitext(self.fp.VALID32_ORI1)[0]
            testpref = os.path.splitext(self.fp.TEST32_ORI1)[0]
        else:
            if train_mode == 'pretrain':
                trainpref = os.path.splitext(self.fp.DAE_ORI1)[0]
            elif train_mode == 'train':
                trainpref = os.path.splitext(self.fp.TRAIN_ORI1)[0]

            elif train_mode == 'trainbig':
                trainpref = os.path.splitext(self.fp.TRAIN_BIG_ORI1)[0]

            elif train_mode == 'finetune':  # finetune
                trainpref = os.path.splitext(self.fp.FINETUNE_ORI1)[0]
            elif train_mode == 'learnfrommistakes':
                trainpref = os.path.splitext(self.fp.TRAIN_LFM_ORI1)[0]
            elif train_mode == 'learnfrommistakes_copy':
                trainpref = os.path.splitext(self.fp.TRAIN_LFM_ORI1)[0]
            elif train_mode == 'train_learnfrommistakes':
                trainpref = os.path.splitext(self.fp.TRAIN_LFM_ORI1)[0]
                testpref = os.path.splitext(self.fp.TEST_ORI1)[0]
            elif train_mode == 'train_learnfrommistakes_2nd142':
                trainpref = os.path.splitext(self.fp.TRAIN_LFM2ND142_ORI1)[0]
                testpref = os.path.splitext(self.fp.TEST_ORI1)[0]
            elif train_mode == 'train_learnfrommistakes_bs204':
                trainpref = os.path.splitext(self.fp.TRAIN_LFM2ND142_ORI1)[0]
                testpref = os.path.splitext(self.fp.TEST_ORI1)[0]
            elif train_mode == 'cvtefinetune':
                trainpref = os.path.splitext(self.fp.CVTE_DEV_ORI1)[0]
            elif train_mode == 'cvtebigfinetune':
                trainpref = os.path.splitext(self.fp.CVTE_TRAIN_ORI1)[0]
                validpref = os.path.splitext(self.fp.CVTE_DEV_ORI1)[0]
                testpref = os.path.splitext(self.fp.CVTE_TEST_ORI1)[0]
            elif train_mode == 'cvtetrain':
                trainpref = os.path.splitext(self.fp.CVTE_TRAIN_ORI1)[0]
                validpref = os.path.splitext(self.fp.CVTE_DEV_ORI1)[0]
                testpref = os.path.splitext(self.fp.CVTE_TEST_ORI1)[0]
            elif train_mode == 'cvtefinetune2':
                trainpref = os.path.splitext(self.fp.CVTE_FINETUNE_ORI1)[0]
                validpref = os.path.splitext(self.fp.CVTE_DEV_ORI1)[0]
                testpref = os.path.splitext(self.fp.CVTE_TEST_ORI1)[0]
            elif train_mode == 'lfm_1st-copy-small-lfm-bpe50-ckpt_best':
                trainpref = os.path.splitext('/container_data/src/helo_word-master/track1/parallel/train_lfm3rd_nois.ori')[0]
            elif train_mode == 'lfm_2nd_lfm142_t2t_bpe50_ckpt148':
                trainpref = os.path.splitext('/container_data/src/helo_word-master/track1/parallel/train_t2t_lfm3rd_nois.ori')[0]
            elif train_mode == 'lfm_1st_base-big_bpe50_ckpt92':
                trainpref = os.path.splitext('/container_data/src/helo_word-master/track1/parallel/train_base_big_lfm1st_nois.ori')[0]

            elif train_mode == 'lfm_1st_base_samll_bpe60_ckpt187':
                trainpref = os.path.splitext('/container_data/src/helo_word-master/track1/parallel/train_base_samll_bpe60_lfm1st.ori')[0]
            elif train_mode == 'lfm_2nd_t2t_bpe60_ckpt220':
                trainpref = os.path.splitext('/container_data/src/helo_word-master/track1/parallel/train_t2t_bpe60_lfm1st.ori')[0]
            elif train_mode == 'lfm_1st_base_small_ckpt29_2':
                trainpref = os.path.splitext('/container_data/src/helo_word-master/track1/parallel/train_base_small_lfm1st.ori')[0]
            elif train_mode == 'lfm_2ndCopy_ckpt176':
                trainpref = os.path.splitext('/container_data/src/helo_word-master/track1/parallel/train_2ndCopy_lfm_176.ori')[0]
            else:
                trainpref = os.path.splitext(self.fp.VALID_ORI1)[0]
            validpref = os.path.splitext(self.fp.VALID_ORI1)[0]
            #testpref = os.path.splitext(self.fp.TEST_ORI1)[0]

        return trainpref, validpref

    def get_subset_datapath(self, subset,bpe_version=50):
        assert subset in self.subsets
        if bpe_version == 32:
            if subset == 'valid':
                gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.m2"
                ori_path = self.fp.WI_DEV_ORI
                ori_bpe_path = None
                gen_subset = "valid"
                scorer_type = 'errant'
        else:
            if subset == 'valid':
                gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.m2"
                ori_path = self.fp.WI_DEV_ORI
                ori_bpe_path = None
                gen_subset = "valid"
                scorer_type = 'errant'

            if subset == 'valid3481':
                gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.4381.m2"
                ori_path = self.fp.WI_DEV_4381_ORI
                ori_bpe_path = None
                gen_subset = "valid"
                scorer_type = 'errant'
            elif subset == 'cvtevalid':
                gold_m2 = f"{self.fp.cvte}/cvte.dev.gold.errant.m2"

                ori_path = self.fp.CVTE_DEV_ORI
                ori_bpe_path = None
                gen_subset = "valid"
                scorer_type = 'errant'


            elif subset == 'train':
                gold_m2 = f"{self.fp.wi_m2}/ABCN.train.gold.bea19.m2"
                ori_path = self.fp.WI_TRAIN_ORI
                ori_bpe_path = None
                gen_subset = "train"
                scorer_type = "errant"

            elif subset == 'trainbig':
                gold_m2 = "/container_data/src/helo_word-master/track1/parallel/trainbig.cor.m2"
                ori_path = "/container_data/src/helo_word-master/track1/parallel/trainbig.ori"
                ori_bpe_path = None
                gen_subset = "train"
                scorer_type = "errant"

            elif subset == 'valid2nd':
                gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.m2"
                ori_path = self.fp.WI_DEV2ND_ORI
                ori_bpe_path = self.fp.WI_DEV2ND_TOK_ORI
                gen_subset = None
                scorer_type = "errant"

            elif subset == 'test':
                gold_m2 = None
                ori_path = self.fp.WI_TEST_ORI
                scorer_type = None
                if bpe_version ==32:
                    ori_bpe_path = self.fp.WI_TEST_TOK32_JEM_ORI
                    gen_subset = None
                elif bpe_version == 60 or bpe_version == 50:
                    #ori_bpe_path =None
                    #ori_path = self.fp.WI_TEST_SP_JEM_ORI
                    ori_bpe_path = self.fp.WI_TEST_TOK_JEM_ORI
                    #gen_subset = "test"
                    gen_subset = None
            elif subset == 'testorg':
                gold_m2 = None
                ori_path = self.fp.WI_TEST_ORI
                scorer_type = None
                if bpe_version ==32:
                    ori_bpe_path = self.fp.WI_TEST_TOK32_ORI
                    gen_subset = None
                elif bpe_version == 60 or bpe_version == 50:
                    #ori_bpe_path =None
                    #ori_path = self.fp.WI_TEST_SP_JEM_ORI
                    ori_bpe_path = self.fp.WI_TEST_TOK_ORI
                    #gen_subset = "test"
                    gen_subset = None
                    
            elif subset == 'soncan':
                gold_m2 = None
                ori_path = "/container_data/sentences_corpus4_use_corr.txt"
                scorer_type = None
                ori_bpe_path = "/container_data/sentences_corpus4_use_corr_tok.txt"
                #gen_subset = "test"
                gen_subset = None
            elif subset == 'devorg':
                gold_m2 = None
                ori_path = self.fp.WI_DEV_ORI
                scorer_type = None
                if bpe_version == 60 or bpe_version == 50:
                    #ori_bpe_path =None
                    gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.m2"
                    #ori_path = self.fp.WI_TEST_SP_JEM_ORI
                    ori_bpe_path = self.fp.WI_DEVNoSp_TOK_ORI
                    #gen_subset = "test"
                    scorer_type = 'errant'
                    gen_subset = None
            elif subset =='onlinetest':
                gold_m2 = None
                ori_path = self.fp.ONLINE_TEST_ORI
                ori_bpe_path = self.fp.ONLINE_TEST_TOK_ORI
                gen_subset = None
                scorer_type = None
            elif subset == 'lwwtest':
                gold_m2 = None
                ori_path = self.fp.WI_TEST_ORI
                scorer_type = None
                if bpe_version ==32:
                    ori_bpe_path = self.fp.WI_TEST_TOK32_NSLWW_ORI
                    gen_subset = None
                elif bpe_version == 50:
                    #ori_bpe_path =None
                    #ori_path = self.fp.WI_TEST_SP_JEM_ORI
                    ori_bpe_path = self.fp.WI_TEST_TOK_NEW_LWW_ORI
                    #gen_subset = "test"
                    gen_subset = None
            elif subset == 'cvte_m2':
                gold_m2 = f"{self.fp.cvte}/CVTE.test.goldm2scorer.m2"
                ori_path = self.fp.CVTE_TEST_ORI
                #gen_subset = "test"
                gen_subset = None

                scorer_type = 'm2scorer'
                if bpe_version ==32:
                    ori_bpe_path = self.fp.CVTE_TEST_TOK32_LWW_ORI
                else:
                    ori_bpe_path = self.fp.CVTE_TEST_TOK_LWW_ORI

            elif subset == 'cvte_errant':
                gold_m2 = f"{self.fp.cvte}/cvte.test.gold.errant.m2"
                ori_path = self.fp.CVTE_TEST_ORI
                gen_subset = "test"
                scorer_type = 'errant'
                if bpe_version ==32:
                    ori_bpe_path = self.fp.CVTE_TEST_TOK32_LWW_ORI
                else:
                    ori_bpe_path = self.fp.CVTE_TEST_TOK_LWW_ORI
            elif subset == 'conll2014':
                if bpe_version == 50:
                    gold_m2 = f"{self.fp.conll2014_m2}/official-2014.combined.m2"
                    ori_path = self.fp.CONLL2014_ORI
                    ori_bpe_path = self.fp.CONLL2014_TOK_ORI
                    gen_subset = None
                    scorer_type = 'm2scorer'

        return gold_m2, ori_path, ori_bpe_path, gen_subset, scorer_type




class Track4(Track):
    def __init__(self):
        super(Track4, self).__init__(4)

    train_modes = ['pretrain', 'train', 'finetune','pretrain', 'dev','pretrain_test','finetunesplit','lfm_1st_copy_el_att_ckpt107']
    subsets = ['valid', 'test', 'conll2014','valid4382','validbpe']

    def get_pref(self, train_mode,bpe_version=50):
        assert train_mode in self.train_modes
        if train_mode == 'pretrainjj':
            trainpref = os.path.splitext(self.fp.DAE_ORI4)[0]
        elif train_mode == 'pretrain':
            trainpref = os.path.splitext(self.fp.DAE_ORI4)[0]
        elif train_mode == 'train':
            trainpref = os.path.splitext(self.fp.TRAIN_ORI4)[0]
        elif train_mode == 'finetune':  # finetune
            trainpref = os.path.splitext(self.fp.FINETUNE_ORI4)[0]
        elif train_mode == 'finetunesplit':  # finetunesplit
            trainpref = os.path.splitext(self.fp.FINETUNE_SPLIT_ORI4)[0]
        elif train_mode == 'pretrain_test':  # finetune
            trainpref = os.path.splitext(self.fp.FINETUNE_ORI4)[0]

        elif train_mode == 'lfm_1st_copy_el_att_ckpt107':
            trainpref = os.path.splitext('/container_data/src/helo_word-master/track4/parallel/train_copy_el_att_lfm1st_nois.ori')[0]
        else:
            trainpref = os.path.splitext(self.fp.VALID_ORI4)[0]
        validpref = os.path.splitext(self.fp.VALID_ORI4)[0]
        #testpref = os.path.splitext(self.fp.TEST_ORI4)[0]
        return trainpref, validpref

    def get_subset_datapath(self, subset,bpe_version=50):
        assert subset in self.subsets

        if subset == 'valid':
            gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.m2"
            ori_path = self.fp.WI_DEV_ORI
            ori_bpe_path = None
            gen_subset = "valid"
            scorer_type = 'errant'
        elif subset == 'valid4382':
            gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.4382.m2"
            ori_path = self.fp.WI_DEV_4382_ORI
            ori_bpe_path = None
            gen_subset = "valid"
            scorer_type = 'errant'

        elif subset == 'validbpe':
            gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.4382.m2"
            #gold_m2 = None
            ori_path = self.fp.WI_DEV_4382_ORI
            ori_bpe_path = None
            #ori_bpe_path = "/container_data/src/helo_word-master/data/parallel/tok/wi.dev4382.tok.ori"
            gen_subset = "valid"
            scorer_type = 'errant'

        elif subset == 'train':
            gold_m2 = f"{self.fp.wi_m2}/ABCN.train.gold.bea19.m2"
            ori_path = self.fp.WI_TRAIN_ORI
            ori_bpe_path = None
            gen_subset = "train"
            scorer_type = "errant"

        elif subset == 'test':
            gold_m2 = None
            ori_path = self.fp.WI_TEST_ORI
            scorer_type = None
            if bpe_version ==32:
                ori_bpe_path = self.fp.WI_TEST_TOK32_JEM_ORI
                gen_subset = test
            else:
                ori_bpe_path =None
                #ori_path = self.fp.WI_TEST_SP_JEM_ORI
                #ori_bpe_path = self.fp.WI_TEST_TOK_JEM_ORI
                gen_subset = "test"
                #gen_subset = None


        else:  # 'conll2014':
            gold_m2 = f"{self.fp.conll2014_m2}/official-2014.combined.m2"
            ori_path = self.fp.CONLL2014_ORI
            ori_bpe_path = self.fp.CONLL2014_TOK_ORI
            gen_subset = None
            scorer_type = 'm2scorer'

        return gold_m2, ori_path, ori_bpe_path, gen_subset, scorer_type


class Track5(Track):
    def __init__(self):
        super(Track5, self).__init__(5)

    train_modes = ['pretrain', 'train', 'finetune', 'dev']
    subsets = ['valid', 'test', 'conll2014']

    def get_pref(self, train_mode):
        assert train_mode in self.train_modes
        if train_mode == 'pretrain':
            trainpref = os.path.splitext(self.fp.DAE_ORI5)[0]
        elif train_mode == 'train':
            trainpref = os.path.splitext(self.fp.TRAIN_ORI5)[0]
        elif train_mode == 'finetune':  # finetune
            trainpref = os.path.splitext(self.fp.FINETUNE_ORI5)[0]
        else:
            trainpref = os.path.splitext(self.fp.VALID_ORI5)[0]
        validpref = os.path.splitext(self.fp.VALID_ORI5)[0]
        return trainpref, validpref

    def get_subset_datapath(self, subset):
        assert subset in self.subsets

        if subset == 'valid':
            gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.m2"
            ori_path = self.fp.WI_DEV_ORI
            ori_bpe_path = None
            gen_subset = "valid"
            scorer_type = 'errant'

        elif subset == 'test':
            gold_m2 = None
            ori_path = self.fp.WI_TEST_ORI
            ori_bpe_path = self.fp.WI_TEST_TOK_ORI
            gen_subset = None
            scorer_type = None

        else:  # 'conll2014':
            gold_m2 = f"{self.fp.conll2014_m2}/official-2014.combined.m2"
            ori_path = self.fp.CONLL2014_ORI
            ori_bpe_path = self.fp.CONLL2014_TOK_ORI
            gen_subset = None
            scorer_type = 'm2scorer'

        return gold_m2, ori_path, ori_bpe_path, gen_subset, scorer_type

class Track3(Track):
    def __init__(self):
        super(Track3, self).__init__(3)

    train_modes = ['pretrain', 'finetune']
    subsets = ['valid', 'test', 'conll2014']

    def get_pref(self, train_mode):
        assert train_mode in self.train_modes
        if train_mode == 'pretrain':
            trainpref = os.path.splitext(self.fp.DAE_ORI3)[0]
        else:
            trainpref = os.path.splitext(self.fp.FINETUNE_ORI3)[0]
        validpref = os.path.splitext(self.fp.VALID_ORI3)[0]
        return trainpref, validpref

    def get_subset_datapath(self, subset):
        assert subset in self.subsets

        if subset == 'valid':
            gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.1k.m2"
            ori_path = self.fp.WI_DEV_1K_ORI
            ori_bpe_path = None
            gen_subset = "valid"
            scorer_type = 'errant'

        elif subset == 'test':
            gold_m2 = None
            ori_path = self.fp.WI_TEST_ORI
            ori_bpe_path = self.fp.WI_TEST_TOK_ORI
            gen_subset = None
            scorer_type = None

        else:  # 'conll2014':
            gold_m2 = f"{self.fp.conll2014_m2}/official-2014.combined.m2"
            ori_path = self.fp.CONLL2014_ORI
            ori_bpe_path = self.fp.CONLL2014_TOK_ORI
            gen_subset = None
            scorer_type = 'm2scorer'

        return gold_m2, ori_path, ori_bpe_path, gen_subset, scorer_type
