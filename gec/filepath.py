import os


class Path:

    # set your root path here
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # hunspell
    aff = f"{root}/data/language_model/dicts/en_AU.aff"
    dic = f"{root}/data/language_model/dicts/en_wiki_rev.dic"

    # capital word dict
    cap_word_dic = f"{root}/data/language_model/dicts/cap_words_dic"

    # lm
    lm_path = f"{root}/data/language_model/wiki103.pt"
    lm_databin = f"{root}/data/language_model/data-bin"
    lm_path2 = f"{root}/data/language_model/wmt19.en/model.pt"
    lm_databin2 = f"{root}/data/language_model/wmt19.en"

    # postprocess
    parallel_to_m2 = f"{root}/errant/parallel_to_m2.py"

    # scorer
    errant = f"{root}/errant/compare_m2.py"
    m2scorer = f"{root}/m2scorer/m2scorer"


class FilePath(object):
    '''define file and directory names and their paths
       lowercase variables are directories, and uppercase ones are files.'''
    def __init__(self):

        # set your root path here
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.gutenberg = f"{self.root}/data/gutenberg"
        self.pbe_traing_data = f"{self.root}/data/training_data_bpe"
        self.tatoeba = f"{self.root}/data/tatoeba"
        self.wiki103 = f"{self.root}/data/wiki103"
        self.for_debugging = f"{self.root}/data/debug_data"
        self.one_billion_words = f"{self.root}/data/one_billion_words"
        self.bea19 = f'{self.root}/data/bea19'
        self.conll2013 = f'{self.root}/data/conll2013'
        self.conll2014 = f'{self.root}/data/conll2014'
        self.cvte = f'{self.root}/data/cvte'

        self.jfleg = f'{self.root}/data/jfleg'
        self.bpe_model = f'{self.root}/data/bpe-model'
        self.bpe_model32k = f'{self.root}/data/bpe-model32k'
        self.parallel = f'{self.root}/data/parallel'
        self.fce = f'{self.bea19}/fce'
        self.wi = f'{self.bea19}/wi+locness'

        self.fce_m2 = f'{self.fce}/m2'
        self.lang8_m2 = f'{self.bea19}/lang8.bea19'
        self.nucle_m2 = f'{self.bea19}/nucle3.3/bea2019'
        self.wi_m2 = f'{self.wi}/m2'
        self.cvte_m2 = f'{self.cvte}/m2'


        self.conll2013_m2 = f'{self.conll2013}/release2.3.1/revised/data'
        self.conll2014_m2 = f'{self.conll2014}/conll14st-test-data/noalt'

        # for dae
        self.GUTENBERG_TXT = f"{self.gutenberg}/gutenberg.txt"
        self.GUTENBERG_CLEAN_TXT = f"{self.gutenberg}/gutenberg_clean.txt"

        self.TATOEBA_TXT = f"{self.tatoeba}/tatoeba.txt"
        self.TATOEBA_CLEAN_TXT = f"{self.tatoeba}/tatoeba_clean.txt"

        self.WIKI103_TXT = f"{self.wiki103}/wiki103.txt"
        self.WIKI103_CLEAN_TXT = f"{self.wiki103}/wiki103_clean.txt"

        self.DEBUG_TXT = f"{self.for_debugging}/deb_data.txt"
        self.ONE_BILLION_WORDS_TXT = f"{self.one_billion_words}/one_billion_words.txt"
        self.ONE_BILLION_WORDS_CLEAN_TXT = f"{self.one_billion_words}/one_billion_words_clean.txt"

        # bpe
        self.BPE_TRAINING_DATA = f"{self.pbe_traing_data}/train_data_bpe.txt"
        self.BPE_MODEL = f"{self.bpe_model}/gutenberg.model"
        self.BPE_MODEL32 = f"{self.bpe_model32k}/gutenberg.model"

        self.BPE_VOCAB = f"{self.bpe_model}/gutenberg.vocab"
        self.BPE_VOCAB_32k = f"{self.bpe_model32k}/gutenberg.vocab"
        # dae tok
        self.GUTENBERG_ORI1 = f"{self.parallel}/tok/gutenberg.tok.wi.train2.ori"
        self.GUTENBERG_COR1 = f"{self.parallel}/tok/gutenberg.tok.wi.train2.cor"
        self.TATOEBA_ORI1 = f"{self.parallel}/tok/tatoeba.tok.wi.train2.ori"
        self.TATOEBA_COR1 = f"{self.parallel}/tok/tatoeba.tok.wi.train2.cor"
        self.WIKI103_ORI1 = f"{self.parallel}/tok/wiki103.tok.wi.train2.ori"
        self.WIKI103_COR1 = f"{self.parallel}/tok/wiki103.tok.wi.train2.cor"

        self.GUTENBERG_CLEAN_ORI1 = f"{self.parallel}/tok/gutenberg.clean.tok.wi.train2.ori"
        self.GUTENBERG_CLEAN_COR1 = f"{self.parallel}/tok/gutenberg.clean.tok.wi.train2.cor"
        self.TATOEBA_CLEAN_ORI1 = f"{self.parallel}/tok//tatoeba.clean.tok.wi.train2.ori"
        self.TATOEBA_CLEAN_COR1 = f"{self.parallel}/tok/tatoeba.clean.tok.wi.train2.cor"
        self.WIKI103_CLEAN_ORI1 = f"{self.parallel}/tok/wiki103.clean.tok.wi.train2.ori"
        self.WIKI103_CLEAN_COR1 = f"{self.parallel}/tok/wiki103.clean.tok.wi.train2.cor"


        self.ONEBILLIONWORDS_ORI1 = f"{self.parallel}/tok/onebillion.tok.wi.train2.ori"
        self.ONEBILLIONWORDS_COR1 = f"{self.parallel}/tok/onebillion.tok.wi.train2.cor"


        self.GUTENBERG_ORI3 = f"{self.parallel}/tok/gutenberg.tok.wi.dev.3k.ori"
        self.GUTENBERG_COR3 = f"{self.parallel}/tok/gutenberg.tok.wi.dev.3k.cor"
        self.TATOEBA_ORI3 = f"{self.parallel}/tok/tatoeba.tok.wi.dev.3k.ori"
        self.TATOEBA_COR3 = f"{self.parallel}/tok/tatoeba.tok.wi.dev.3k.cor"
        self.WIKI103_ORI3 = f"{self.parallel}/tok/wiki103.tok.wi.dev.3k.ori"
        self.WIKI103_COR3 = f"{self.parallel}/tok/wiki103.tok.wi.dev.3k.cor"

        self.GUTENBERG_ORI0 = f"{self.parallel}/tok/gutenberg.tok.nucle.ori"
        self.GUTENBERG_COR0 = f"{self.parallel}/tok/gutenberg.tok.nucle.cor"
        self.TATOEBA_ORI0 = f"{self.parallel}/tok/tatoeba.tok.nucle.ori"
        self.TATOEBA_COR0 = f"{self.parallel}/tok/tatoeba.tok.nucle.cor"
        self.WIKI103_ORI0 = f"{self.parallel}/tok/wiki103.tok.nucle.ori"
        self.WIKI103_COR0 = f"{self.parallel}/tok/wiki103.tok.nucle.cor"

        # raw
        self.FCE_ORI = f"{self.parallel}/raw/fce.ori"
        self.FCE_COR = f"{self.parallel}/raw/fce.cor"
        self.LANG8_ORI = f"{self.parallel}/raw/lang8.ori"
        self.LANG8_COR = f"{self.parallel}/raw/lang8.cor"
        self.NUCLE_ORI = f"{self.parallel}/raw/nucle.ori"
        self.NUCLE_COR = f"{self.parallel}/raw/nucle.cor"
        self.WI_TRAIN_ORI = f"{self.parallel}/raw/wi.train.ori"
        self.WI_DEV2ND_ORI = "/container_data/src/helo_word-master/track1/outputs60/bestckpt50_60/1stcopy6ckpt174.wi.dev.cor"

        self.WI_ALL_TRAIN_ORI = f"{self.parallel}/raw/wi.all.train.ori"

        self.WI_TRAIN_GEN_ORI = f"{self.parallel}/raw/wi.train.gen.ori"
        self.WI_TRAIN_GEN_COR = f"{self.parallel}/raw/wi.train.gen.cor"

        self.WI_LFM_TRAIN_ORI = "/container_data/src/helo_word-master/data/parallel/raw/1st-copy-small_bpe50_ckpt66.wi.train.cor"
        self.WI_LFM2ND142_TRAIN_ORI = "/container_data/src/helo_word-master/track1/outputs50/bestckp50/1st_t2t_lfm_bpe50-ckpt142.wi.train.cor"
        self.WI_LFMBS204_TRAIN_ORI = "/container_data/src/helo_word-master/track1/outputs50/bestckp50/1st_base-small_bpe50_ckpt204.wi.train.cor"

        #self.WI_LFM_TRAIN_COR = f"{self.parallel}/raw/wi.train.gen.cor"

        self.WI_VALID_GEN_ORI = f"{self.parallel}/raw/wi.valid.gen.ori"
        self.WI_TEST_GEN_ORI = f"{self.parallel}/raw/wi.test.gen.ori"


        self.WI_TRAIN_COR = f"{self.parallel}/raw/wi.train.cor"

        self.WI_DEV_ORI = f"{self.parallel}/raw/wi.dev.ori"
        self.WI_DEV_4382_ORI = f"{self.parallel}/raw/wi.dev4382.ori"
        self.WI_DEV_4382_COR = f"{self.parallel}/raw/wi.dev4382.cor"



        self.WI_DEV_COR = f"{self.parallel}/raw/wi.dev.cor"
        self.WI_TEST_ORI = f"{self.parallel}/raw/ABCN.test.bea19.orig"
        self.ONLINE_TEST_ORI = f"{self.parallel}/raw/online.test.ori"

        self.CVTE_TEST_ORI = f"{self.parallel}/raw/cvte.test.ori"
        self.CVTE_TEST_COR = f"{self.parallel}/raw/cvte.test.cor"

        self.CVTE_DEV_ORI = f"{self.parallel}/raw/cvte.dev.ori"
        self.CVTE_DEV_COR = f"{self.parallel}/raw/cvte.dev.cor"

        self.CVTE_TRAIN_ORI = f"{self.parallel}/raw/cvte.train.ori"
        self.CVTE_TRAIN_COR = f"{self.parallel}/raw/cvte.train.cor"

        self.WI_TEST_BESTMODEL_ORI = f"{self.parallel}/raw/wi.test.ori.bestmodel"
        # self.WI_TEST_COR = f"{self.parallel}/raw/wi.test.cor"

        self.WI_DEV_3K_ORI = f"{self.parallel}/raw/wi.dev.3k.ori"
        self.WI_DEV_3K_COR = f"{self.parallel}/raw/wi.dev.3k.cor"
        self.WI_DEV_1K_ORI = f"{self.parallel}/raw/wi.dev.1k.ori"
        self.WI_DEV_1K_COR = f"{self.parallel}/raw/wi.dev.1k.cor"

        self.CONLL2013_ORI = f"{self.parallel}/raw/conll2013.ori"
        self.CONLL2013_COR = f"{self.parallel}/raw/conll2013.cor"
        self.CONLL2014_ORI = f"{self.parallel}/raw/conll2014.ori"
        self.CONLL2014_COR = f"{self.parallel}/raw/conll2014.cor"
        self.JFLEG_ORI = f"{self.jfleg}/test/test.src"

        # sp
        self.FCE_SP_ORI = f"{self.parallel}/sp/fce.sp.ori"
        self.LANG8_SP_ORI = f"{self.parallel}/sp/lang8.sp.ori"
        self.NUCLE_SP_ORI = f"{self.parallel}/sp/nucle.sp.ori"
        self.WI_TRAIN_SP_ORI = f"{self.parallel}/sp/wi.train.sp.ori"
        self.CVTE_TRAIN_SP_ORI = f"{self.parallel}/sp/cvte.train.sp.ori"
        self.CVTE_TEST_SP_ORI = f"{self.parallel}/sp/cvte.test.sp.ori"


        self.WI_DEV_SP_ORI = f"{self.parallel}/sp/wi.dev.sp.ori"
        self.WI_DEV_SP_NSLWW_ORI = f"{self.parallel}/sp/wi.dev.sp.ori.nlww"

        self.WI_TEST_SP_ORI = f"{self.parallel}/sp/cvte.test.sp.ori"
        self.CVTE_TEST_SP_ORI = f"{self.parallel}/sp/cvte.test.sp.ori"
        self.CVTE_TEST_SP_LWW_ORI = f"{self.parallel}/sp/cvte.test.sp.ori.lww"


        self.WI_TEST_SP_JEM_ORI = f"{self.parallel}/sp/wi.test.sp.ori.jem"
        self.WI_TEST_SP_NEW_LWW_ORI = f"{self.parallel}/sp/wi.test.sp.ori.newlww"
        self.WI_TEST_SP_MODEL_OUT_ORI = f"{self.parallel}/raw/4th_base_ck151.ABCN.test.bea19-unk-rerank-spell-max-10.cor"

        self.WI_DEV_3K_SP_ORI = f"{self.parallel}/sp/wi.dev.3k.sp.ori"
        self.WI_DEV_1K_SP_ORI = f"{self.parallel}/sp/wi.dev.1k.sp.ori"

        self.CONLL2013_SP_ORI = f"{self.parallel}/sp/conll2013.sp.ori"
        self.CONLL2014_SP_ORI = f"{self.parallel}/sp/conll2014.sp.ori"
        self.JFLEG_SP_ORI = f"{self.parallel}/sp/jfleg.sp.ori"

        # tok
        self.FCE_TOK_ORI = f"{self.parallel}/tok/fce.tok.ori"
        self.FCE_TOK_COR = f"{self.parallel}/tok/fce.tok.cor"
        self.LANG8_TOK_ORI = f"{self.parallel}/tok/lang8.tok.ori"
        self.LANG8_TOK_COR = f"{self.parallel}/tok/lang8.tok.cor"
        self.NUCLE_TOK_ORI = f"{self.parallel}/tok/nucle.tok.ori"
        self.NUCLE_TOK_COR = f"{self.parallel}/tok/nucle.tok.cor"
        self.WI_TRAIN_TOK_ORI = f"{self.parallel}/tok/wi.train.tok.ori"
        self.WI_TRAIN_TOK_GEN_ORI = f"{self.parallel}/tok/wi.train.tok.gen.ori"
        self.WI_TRAIN_TOK32_GEN_ORI = f"{self.parallel}/tok/wi.train.tok32.gen.ori"
        self.WI_TRAIN_TOK32_GEN_COR = f"{self.parallel}/tok/wi.train.tok32.gen.cor"

        self.WI_TEST_TOK_GEN_ORI = f"{self.parallel}/tok/wi.test.tok32.gen.ori"
        self.WI_VALID_TOK_GEN_ORI = f"{self.parallel}/tok/wi.valid.tok32.gen.ori"
        self.WI_VALID_TOK32_ORI = f"{self.parallel}/tok/wi.valid.tok32.ori"
        self.WI_VALID_TOK32_NSLWW_ORI = f"{self.parallel}/tok/wi.valid.tok32.nslww.ori"

        self.WI_VALID_TOK32_COR = f"{self.parallel}/tok/wi.valid.tok32.cor"

        self.WI_TRAIN_TOK_COR = f"{self.parallel}/tok/wi.train.tok.cor"
        self.WI_TRAIN_TOK32_COR = f"{self.parallel}/tok32/wi.train.tok.cor"
        self.WI_TRAIN_TOK32_ORI = f"{self.parallel}/tok32/wi.train.tok.ori"

        self.WI_DEV_TOK_ORI = f"{self.parallel}/tok/wi.dev.tok.ori"
        self.WI_DEVNoSp_TOK_ORI = f"{self.parallel}/tok/wi.devnosp.tok.ori"

        self.WI_DEV_TOK32_ORI = f"{self.parallel}/tok32/wi.dev.tok.ori"

        self.WI_DEV_TOK_COR = f"{self.parallel}/tok/wi.dev.tok.cor"
        self.WI_DEV_TOK32_COR = f"{self.parallel}/tok32/wi.dev.tok.cor"

        self.WI_DEV_NTOK_ORI = f"{self.parallel}/tok/wi.dev.ntok.ori"
        self.WI_DEV_NTOK_COR = f"{self.parallel}/tok/wi.dev.ntok.cor"

        self.WI_TEST_TOK_BEST_COR = f"{self.parallel}/tok/wi.test.tok.ori.bestmodel.cor"
        self.WI_TEST_BEST_COR = f"{self.parallel}/raw/wi.test.ori.bestmodel.cor"

        self.WI_TEST_TOK32_JEM_ORI = f"{self.parallel}/tok/wi.test.tok32.ori.jem"
        self.WI_TEST_TOK_ORI = f"{self.parallel}/tok/wi.test.tok.ori"
        self.WI_DEV2ND_TOK_ORI = f"{self.parallel}/tok/wi.dev2nd.tok.ori"

        self.WI_TEST_TOK32_NSLWW_ORI = f"{self.parallel}/tok/wi.test.tok32.nslww.ori"
        self.WI_TEST_TOK_NEW_LWW_ORI = f"{self.parallel}/tok/wi.test.tok.newlww.ori"


        self.CVTE_TEST_TOK32_ORI = f"{self.parallel}/tok/cvte.test.tok32.ori"
        self.CVTE_TEST_TOK32_LWW_ORI = f"{self.parallel}/tok/cvte.test.tok32.lww.ori"

        self.CVTE_TEST_TOK_ORI = f"{self.parallel}/tok/cvte.test.tok.ori"
        self.CVTE_DEV_TOK_ORI = f"{self.parallel}/tok/cvte.dev.tok.ori"
        self.CVTE_DEV_TOK_COR = f"{self.parallel}/tok/cvte.dev.tok.cor"

        self.CVTE_TRAIN_SP_TOK_ORI = f"{self.parallel}/tok/cvte.train.sp.tok.ori"
        self.CVTE_TRAIN_TOK_ORI = f"{self.parallel}/tok/cvte.train.tok.ori"

        self.CVTE_TRAIN_TOK_COR = f"{self.parallel}/tok/cvte.train.tok.cor"

        self.CVTE_TEST_TOK_LWW_ORI = f"{self.parallel}/tok/cvte.test.tok.lww.ori"

        self.CVTE_TEST_TOK32_COR = f"{self.parallel}/tok/cvte.test.tok32.cor"
        self.CVTE_TEST_TOK_COR = f"{self.parallel}/tok/cvte.test.tok.cor"

        self.WI_TEST_TOK32_MODEL_OUT_ORI = f"{self.parallel}/tok/wi.test.tok32.model.out.ori"

        self.WI_TEST_TOK_JEM_ORI = f"{self.parallel}/tok/wi.test.tok.jem.ori"
        self.ONLINE_TEST_TOK_ORI = f"{self.parallel}/tok/online.test.tok.ori"

        #self.WI_TEST_TOK_JEM_COR = f"{self.parallel}/tok/wi.test.tok.cor.bestmodel"


        self.WI_TEST_NTOK_ORI = f"{self.parallel}/tok/wi.test.ntok.ori"


        # self.WI_TEST_TOK_COR = f"{self.parallel}/tok/wi.test.tok.cor"
        self.WI_LFM_TRAIN_TOK_ORI = f"{self.parallel}/tok/wi.lfm.train.tok.ori"
        self.WI_LFM_TRAIN_TOK_COR = f"{self.parallel}/tok/wi.lfm.train.tok.cor"

        self.WI_LFM2ND142_TRAIN_TOK_ORI = f"{self.parallel}/tok/wi.lfm2nd142.train.tok.ori"
        self.WI_LFM2ND142_TRAIN_TOK_COR = f"{self.parallel}/tok/wi.lfm2nd142.train.tok.cor"

        self.WI_LFMBS204_TRAIN_TOK_ORI = f"{self.parallel}/tok/wi.lfmbs204.train.tok.ori"
        self.WI_LFMBS204_TRAIN_TOK_COR = f"{self.parallel}/tok/wi.lfmbs204.train.tok.cor"


        self.WI_DEV_3K_TOK_ORI = f"{self.parallel}/tok/wi.dev.3k.tok.ori"
        self.WI_DEV_3K_TOK_COR = f"{self.parallel}/tok/wi.dev.3k.tok.cor"
        self.WI_DEV_1K_TOK_ORI = f"{self.parallel}/tok/wi.dev.1k.tok.ori"
        self.WI_DEV_1K_TOK_COR = f"{self.parallel}/tok/wi.dev.1k.tok.cor"

        self.CONLL2013_TOK_ORI = f"{self.parallel}/tok/conll2013.tok.ori"
        self.CONLL2013_TOK_COR = f"{self.parallel}/tok/conll2013.tok.cor"
        self.CONLL2014_TOK_ORI = f"{self.parallel}/tok/conll2014.tok.ori"
        self.CONLL2014_TOK_COR = f"{self.parallel}/tok/conll2014.tok.cor"
        self.JFLEG_TOK_ORI = f"{self.parallel}/tok/jfleg.tok.ori"
        # self.JFLEG_TOK_COR = f"{self.parallel}/tok/jfleg.tok.cor"

        # labels
        self.FCE_WITH_LABELS_ORI = f"{self.parallel}/lbs/fce.lbs.ori"
        self.FCE_WITH_LABELS_COR = f"{self.parallel}/lbs/fce.lbs.cor"
        self.LANG8_WITH_LABELS_ORI = f"{self.parallel}/lbs/lang8.lbs.ori"
        self.LANG8_WITH_LABELS_COR = f"{self.parallel}/lbs/lang8.lbs.cor"
        self.NUCLE_WITH_LABELS_ORI = f"{self.parallel}/lbs/nucle.lbs.ori"
        self.NUCLE_WITH_LABELS_COR = f"{self.parallel}/lbs/nucle.lbs.cor"
        self.WI_TRAIN_WITH_LABELS_ORI = f"{self.parallel}/lbs/wi.train.lbs.ori"
        self.WI_TRAIN_WITH_LABELS_COR = f"{self.parallel}/lbs/wi.train.lbs.cor"
        self.WI_DEV_WITH_LABELS_ORI = f"{self.parallel}/lbs/wi.dev.lbs.ori"
        self.WI_DEV_WITH_LABELS_COR = f"{self.parallel}/lbs/wi.dev.lbs.cor"
        self.WI_TEST_WITH_LABELS_ORI = f"{self.parallel}/lbs/wi.test.lbs.ori"
        # self.WI_TEST_TOK_COR = f"{self.parallel}/tok/wi.test.tok.cor"

        self.WI_DEV_3K_TOK_ORI = f"{self.parallel}/tok/wi.dev.3k.tok.ori"
        self.WI_DEV_3K_TOK_COR = f"{self.parallel}/tok/wi.dev.3k.tok.cor"
        self.WI_DEV_1K_TOK_ORI = f"{self.parallel}/tok/wi.dev.1k.tok.ori"
        self.WI_DEV_1K_TOK_COR = f"{self.parallel}/tok/wi.dev.1k.tok.cor"

        self.CONLL2013_TOK_ORI = f"{self.parallel}/tok/conll2013.tok.ori"
        self.CONLL2013_TOK_COR = f"{self.parallel}/tok/conll2013.tok.cor"
        self.CONLL2014_TOK_ORI = f"{self.parallel}/tok/conll2014.tok.ori"
        self.CONLL2014_TOK_COR = f"{self.parallel}/tok/conll2014.tok.cor"
        self.JFLEG_TOK_ORI = f"{self.parallel}/tok/jfleg.tok.ori"
        # self.JFLEG_TOK_COR = f"{self.parallel}/tok/jfleg.tok.cor"

        # track1
        self.DAE_ORI1 = f"{self.root}/track1/parallel/dae.ori"
        self.DAE_COR1 = f"{self.root}/track1/parallel/dae.cor"
        self.TRAIN_ORI1 = f"{self.root}/track1/parallel/train.ori"
        self.TRAIN_COR1 = f"{self.root}/track1/parallel/train.cor"

        self.TRAIN_BIG_ORI1 = f"{self.root}/track1/parallel/trainbig.ori"
        self.TRAIN_BIG_COR1 = f"{self.root}/track1/parallel/trainbig.cor"


        self.DAE32_ORI1 = f"{self.root}/track1/parallel32/dae.ori"
        self.DAE32_COR1 = f"{self.root}/track1/parallel32/dae.cor"
        self.TRAIN32_ORI1 = f"{self.root}/track1/parallel32/train.ori"
        self.TRAIN32_COR1 = f"{self.root}/track1/parallel32/train.cor"

        self.FINETUNE_ORI1 = f"{self.root}/track1/parallel/finetune.ori"
        self.FINETUNE_COR1 = f"{self.root}/track1/parallel/finetune.cor"


        self.FINETUNE32_ORI1 = f"{self.root}/track1/parallel32/finetune.ori"
        self.FINETUNE32_COR1 = f"{self.root}/track1/parallel32/finetune.cor"

        self.LEARNFROMMISTAKES32_ORI1 = f"{self.root}/track1/parallel32/gentraintok32.ori"
        self.LEARNFROMMISTAKES32_COR1 = f"{self.root}/track1/parallel32/gentraintok32.cor"


        self.TRAIN_LFM_ORI1 = f"{self.root}/track1/parallel/lfm.ori"
        self.TRAIN_LFM_COR1 = f"{self.root}/track1/parallel/lfm.cor"

        self.TRAIN_LFM2ND142_ORI1 = f"{self.root}/track1/parallel/lfm2nd142.ori"
        self.TRAIN_LFM2ND142_COR1 = f"{self.root}/track1/parallel/lfm2nd142.cor"

        self.TRAIN_LFMBS204_ORI1 = f"{self.root}/track1/parallel/lfmbs204.ori"
        self.TRAIN_LFMBS204_COR1 = f"{self.root}/track1/parallel/lfmbs204.cor"

        self.TRAIN_LFM32_ORI1 = f"{self.root}/track1/parallel32/finetunelfm32.ori"
        self.TRAIN_LFM32_COR1 = f"{self.root}/track1/parallel32/finetunelfm32.cor"

        self.VALID32_ORI1 = f"{self.root}/track1/parallel32/valid.ori"
        self.VALID32_COR1 = f"{self.root}/track1/parallel32/valid.cor"

        self.TEST32_ORI1 = f"{self.root}/track1/parallel32/test.ori"
        self.TEST32_COR1 = f"{self.root}/track1/parallel32/test.cor"

        self.VALID_ORI1 = f"{self.root}/track1/parallel/valid.ori"
        self.VALID_COR1 = f"{self.root}/track1/parallel/valid.cor"

        self.TEST_ORI1 = f"{self.root}/track1/parallel/test.ori"
        self.TEST_COR1 = f"{self.root}/track1/parallel/test.cor"

        self.CVTE_DEV_ORI1 = f"{self.root}/track1/parallel/cvtedev.ori"
        self.CVTE_DEV_COR1 = f"{self.root}/track1/parallel/cvtedev.cor"

        self.CVTE_TEST_ORI1 = f"{self.root}/track1/parallel/cvtetest.ori"
        self.CVTE_TEST_COR1 = f"{self.root}/track1/parallel/cvtetest.cor"

        self.CVTE_TRAIN_ORI1 = f"{self.root}/track1/parallel/cvtetrain.ori"
        self.CVTE_TRAIN_COR1 = f"{self.root}/track1/parallel/cvtetrain.cor"

        self.CVTE_FINETUNE_ORI1 = f"{self.root}/track1/parallel/cvtefinetune.ori"
        self.CVTE_FINETUNE_COR1 = f"{self.root}/track1/parallel/cvtefinetune.cor"

        self.CVTE32_ORI1 = f"{self.root}/track1/parallel32/cvte.ori"
        self.CVTE32_COR1 = f"{self.root}/track1/parallel32/cvte.cor"

        self.LFM_TRAIN_TOK_ORI = f"{self.root}/track1/output/learnfrommistakes.ori"
        self.LFM_TRAIN_TOK_COR = f"{self.root}/track1/output/learnfrommistakes.cor"

        "/container_data/src/helo_word-master/track1/output/learnfrommistakes.ori"
        # track4
        self.DAE_ORI4 = f"{self.root}/track4/parallel/dae.ori"
        self.DAE_COR4 = f"{self.root}/track4/parallel/dae.cor"
        self.TRAIN_ORI4 = f"{self.root}/track4/parallel/train.ori"
        self.TRAIN_COR4 = f"{self.root}/track4/parallel/train.cor"
        self.FINETUNE_ORI4 = f"{self.root}/track4/parallel/finetune.ori"
        self.FINETUNE_COR4 = f"{self.root}/track4/parallel/finetune.cor"
        self.FINETUNE_SPLIT_ORI4 = f"{self.root}/track4/parallel/finetunesplit.ori"
        self.FINETUNE_SPLIT_COR4 = f"{self.root}/track4/parallel/finetunesplit.cor"
        self.VALID_ORI4 = f"{self.root}/track4/parallel/valid.ori"
        self.VALID_COR4 = f"{self.root}/track4/parallel/valid.cor"
        self.TEST_ORI4 = f"{self.root}/track4/parallel/test.ori"
        self.TEST_COR4 = f"{self.root}/track4/parallel/test.cor"

        # track5
        self.DAE_ORI5 = f"{self.root}/track5/parallel/dae.ori"
        self.DAE_COR5 = f"{self.root}/track5/parallel/dae.cor"
        self.TRAIN_ORI5 = f"{self.root}/track5/parallel/train.ori"
        self.TRAIN_COR5 = f"{self.root}/track5/parallel/train.cor"
        self.FINETUNE_ORI5 = f"{self.root}/track5/parallel/finetune.ori"
        self.FINETUNE_COR5 = f"{self.root}/track5/parallel/finetune.cor"
        self.VALID_ORI5 = f"{self.root}/track5/parallel/valid.ori"
        self.VALID_COR5 = f"{self.root}/track5/parallel/valid.cor"
        self.TEST_ORI5 = f"{self.root}/track5/parallel/test.ori"
        self.TEST_COR5 = f"{self.root}/track5/parallel/test.cor"

        # track3
        self.DAE_ORI3 = f"{self.root}/track3/parallel/dae.ori"
        self.DAE_COR3 = f"{self.root}/track3/parallel/dae.cor"
        self.FINETUNE_ORI3 = f"{self.root}/track3/parallel/finetune.ori"
        self.FINETUNE_COR3 = f"{self.root}/track3/parallel/finetune.cor"
        self.VALID_ORI3 = f"{self.root}/track3/parallel/valid.ori"
        self.VALID_COR3 = f"{self.root}/track3/parallel/valid.cor"
        self.TEST_ORI3 = f"{self.root}/track3/parallel/test.ori"
        self.TEST_COR3 = f"{self.root}/track3/parallel/test.cor"

        # track0
        self.DAE_ORI0 = f"{self.root}/track0/parallel/dae.ori"
        self.DAE_COR0 = f"{self.root}/track0/parallel/dae.cor"
        self.TRAIN_ORI0 = f"{self.root}/track0/parallel/train.ori"
        self.TRAIN_COR0 = f"{self.root}/track0/parallel/train.cor"
        self.FINETUNE_ORI0 = f"{self.root}/track0/parallel/finetune.ori"
        self.FINETUNE_COR0 = f"{self.root}/track0/parallel/finetune.cor"
        self.VALID_ORI0 = f"{self.root}/track0/parallel/valid.ori"
        self.VALID_COR0 = f"{self.root}/track0/parallel/valid.cor"
        self.TEST_ORI0 = f"{self.root}/track0/parallel/test.ori"
        self.TEST_COR0 = f"{self.root}/track0/parallel/test.cor"
        self.TEST_ORI_JFLEG0 = f"{self.root}/track0/parallel/test_jfleg.ori"
        self.TEST_COR_JFLEG0 = f"{self.root}/track0/parallel/test_jfleg.cor"

        # make_dirs
        #self.make_dirs()

    def make_dirs(self):
        fnames = [getattr(self, attr) for attr in dir(self) if attr[0].isupper()]
        for fname in fnames:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
