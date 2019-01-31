from dataclasses import dataclass
import os
from fastai import *
from fastai.text import *
from sacremoses import MosesTokenizer

URL_TOKEN = 'xxurl'


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
class ExperimentClsData:
    dataset_path: str  # data_dir
    vocab_path: str
    bs: int
    mark_fields: bool = False
    base_lm_path: str = None
    backwards: str = False
    max_vocab: int = 60000

    emb_sz: int = 400
    nh: int = None
    nl: int = 3

    # these hyperparameters are for training on ~100M tokens (e.g. WikiText-103)
    # for training on smaller datasets, more dropout is necessary
    dps = (
        0.25, 0.1, 0.2, 0.02, 0.15)  # consider removing dps & clip from the default hyperparams and put them to train
    clip: float = 0.12
    bptt: int = 70

    # alpha and beta - defaults like in fastai/text/learner.py:RNNLearner()
    rnn_alpha: float = 2  # activation regularization (AR)
    rnn_beta: float = 1  # temporal activation regularization (TAR)

    # preprocessing
    lang: str = 'en'
    pre_rules: Collection[Callable[[str], str]] = tuple(defaults.text_pre_rules)
    post_rules: Collection[Callable[[str], str]] = tuple(defaults.text_post_rules)
    text_cols: IntsOrStrs = 1
    label_cols: IntsOrStrs = 0

    @property
    def cache_dir(self):
        return os.path.join(self.dataset_path, 'models', 'cls_cache')

    def load_cls_data(self, trn_df, val_df, tst_df):
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


@dataclass
class FactCheckingClsData(ExperimentClsData):
    mark_fields: bool = True
    pre_rules: Collection[Callable[[str], str]] = tuple([remove_urls] + defaults.text_pre_rules)
    text_cols: IntsOrStrs = ('category', 'subject', 'body')
    label_cols: IntsOrStrs = ('fact_label',)
