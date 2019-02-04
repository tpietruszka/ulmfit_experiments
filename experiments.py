from dataclasses import dataclass
import os
from fastai import *
from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback
from sacremoses import MosesTokenizer
from sklearn.model_selection import KFold
from . import sequence_aggregations
from . import classifiers

URL_TOKEN = 'xxurl'
PAD_TOKEN_ID = 1


def remove_urls(t: str) -> str:
    return re.sub(r'http\S+', URL_TOKEN, t)


class MosesTokenizerFunc(BaseTokenizer):
    """
    Wrapper around a MosesTokenizer to make it a `BaseTokenizer`.
    From github.com/n-waves/ulmfit-multilingual
    """

    def __init__(self, lang: str):
        super().__init__(lang=lang)
        self.tok = MosesTokenizer(lang)

    def tokenizer(self, t: str) -> List[str]:
        return self.tok.tokenize(t, return_str=False, escape=False)

    def add_special_cases(self, toks: Collection[str]):
        for w in toks:
            assert len(self.tokenizer(w)) == 1, f"Tokenizer is unable to keep {w} as one token!"


@dataclass
class ExperimentCls:
    dataset_path: str  # data_dir
    vocab_path: str
    bs: int
    mark_fields: bool = False
    base_lm_path: str = None
    backwards: str = False
    max_vocab: int = 60000

    # preprocessing
    lang: str = 'en'
    pre_rules: Collection[Callable[[str], str]] = tuple(defaults.text_pre_rules)
    post_rules: Collection[Callable[[str], str]] = tuple(defaults.text_post_rules)
    text_cols: IntsOrStrs = 1
    label_cols: IntsOrStrs = 0

    # model params

    emb_sz: int = 400
    nh: int = 1150
    nl: int = 3
    bptt: int = 70
    max_len: int = 70 * 20
    # these hyperparameters are for training on ~100M tokens (e.g. WikiText-103)
    # for training on smaller datasets, more dropout is necessary
    encoder_dps: Sequence[float] = tuple(default_dropout['classifier'][
                                         :-1])  # TODO: replace with fastai classifier defaults. Originally first 4 for the encoder, last for decoder
    classifier_dps: Sequence[float] = (0.4, 0.1)
    lin_ftrs = (50,)
    drop_mult: float = 1.  # main switch to proportionally rescale dps
    clip: float = 0.12
    true_wd: bool = True

    # alpha and beta - defaults like in fastai/text/learner.py:RNNLearner()
    rnn_alpha: float = 2  # activation regularization (AR)
    rnn_beta: float = 1  # temporal activation regularization (TAR)

    @property
    def cache_dir(self):
        return os.path.join(self.dataset_path, 'models', 'cls_cache')

    def get_data_bunch(self, trn_df, val_df, tst_df) -> DataBunch:
        args = dict(tokenizer=Tokenizer(tok_func=MosesTokenizerFunc, lang=self.lang, pre_rules=self.pre_rules,
                                        post_rules=self.post_rules))

        with open(self.vocab_path, 'rb') as file:
            vocab = Vocab(pickle.load(file))
        args['vocab'] = vocab
        print(f"Running tokenization...")
        data_cls = TextClasDataBunch.from_df(path=self.cache_dir, train_df=trn_df, valid_df=val_df,
                                             test_df=tst_df, max_vocab=self.max_vocab, bs=self.bs,
                                             backwards=self.backwards, text_cols=self.text_cols,
                                             label_cols=self.label_cols, mark_fields=self.mark_fields, **args)
        print('Size of vocabulary:', len(data_cls.vocab.itos))
        print('First 20 words in vocab:', data_cls.vocab.itos[:20])
        return data_cls

    def get_learner(self, data_bunch: DataBunch,
                    agg_model: sequence_aggregations.Aggregation) -> 'TextClassifierLearner':
        encoder_dps = [x * self.drop_mult for x in self.encoder_dps]
        num_classes = data_bunch.c
        vocab_size = len(data_bunch.vocab.itos)
        lin_ftrs = list(self.lin_ftrs) + [num_classes]
        rnn_enc = MultiBatchRNNCore(self.bptt, self.max_len, vocab_size, self.emb_sz, self.nh, self.nl,
                                    pad_token=PAD_TOKEN_ID, input_p=encoder_dps[0], weight_p=encoder_dps[1],
                                    embed_p=encoder_dps[2], hidden_p=encoder_dps[3])
        classifier = classifiers.SequenceAggregatingClassifier(agg_model, lin_ftrs, self.classifier_dps)
        model = SequentialRNN(rnn_enc, classifier)
        learn = RNNLearner(data_bunch, model, self.bptt, split_func=rnn_classifier_split, true_wd=self.true_wd)
        learn.callback_fns += [partial(CSVLogger, filename=f"{learn.model_dir}/cls-history"),
                               partial(SaveModelCallback, every='improvement', name='cls_best')]
        return learn


@dataclass
class FactChecking(ExperimentCls):
    mark_fields: bool = True
    pre_rules: Collection[Callable[[str], str]] = tuple([remove_urls] + defaults.text_pre_rules)
    text_cols: IntsOrStrs = ('category', 'subject', 'body')
    label_cols: IntsOrStrs = ('fact_label',)

    num_cv_splits: int = 5
    cv_random_state: int = 17

    def get_dfs(self, fold_num: int = 0):
        trn_df_full = pd.read_csv(os.path.join(self.dataset_path, 'train.csv'))
        tst_df = pd.read_csv(os.path.join(self.dataset_path, 'test.csv'))

        kf = KFold(self.num_cv_splits, True, random_state=self.cv_random_state)
        split = list(kf.split(trn_df_full))[fold_num]
        trn_df = trn_df_full.iloc[split[0]]
        val_df = trn_df_full.iloc[split[1]]
        return trn_df, val_df, tst_df
