from fastai.text import *
from fastai.callbacks import SaveModelCallback
from sacremoses import MosesTokenizer
from sklearn.model_selection import KFold
from . import sequence_aggregations
from . import classifiers
from . import sentencepiece_tok
from .utils import *
from abc import abstractmethod

URL_TOKEN = 'xxurl'
PAD_TOKEN_ID = 1

CLS_BEST_FILE = 'cls_best'

metrics_registry = {'accuracy': accuracy, 'f1': FBeta(beta=1)}
callbacks_registry = {'auroc': AUROC,
                      'average_precision_score': AveragePrecisionScore}


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
class Fit1CycleParams:
    """
    This class should be used like:
    ```
        params = Fit1CycleParams(-1, 1)
        learn.freeze_to(params.freeze_to)
        learn.fit_one_cycle(**params.to_dict())
    ```
    """
    freeze_to: int
    cyc_len: int
    lr_max_last: float = 1e-3
    lr_last_to_first_ratio: float = (2.6 ** 4)  # lr_max_first == lr_max_last / lr_last_to_first_ratio
    moms: (float, float) = (0.8, 0.7)
    div_factor: float = 25.0
    pct_start: float = 0.3

    def keys(self):
        return ['cyc_len', 'max_lr', 'moms', 'div_factor', 'pct_start']

    def __getitem__(self, item):
        if item == 'max_lr':
            return slice(self.lr_max_last / self.lr_last_to_first_ratio, self.lr_max_last)
        return getattr(self, item)

    def to_dict(self):
        return {k: self[k] for k in self.keys()}


@dataclass
class ExperimentCls(metaclass=RegisteredAbstractMeta, is_registry=True):
    dataset_path: str  # data_dir
    encoder_subdir: str  # directory inside of 'dataset_path' containing vocab, fwd and optionally bwd encoders

    training_phases: List[Dict]  # arguments to Fi1CycleParams()
    aggregation_class: str
    aggregation_params: Dict

    backwards: str = False
    bidir: bool = False

    bs: int = 40
    mark_fields: bool = False
    max_vocab: int = 60000

    # preprocessing
    lang: str = 'en'
    pre_rules: Collection[Callable[[str], str]] = tuple(defaults.text_pre_rules)
    post_rules: Collection[Callable[[str], str]] = tuple(defaults.text_post_rules)
    text_cols: IntsOrStrs = 1
    label_col: IntsOrStrs = 0

    # model params
    emb_sz: int = 400
    nh: int = 1150
    nl: int = 3
    bptt: int = 70
    max_len: int = 70 * 20
    rnn_output_layers: List[int] = (-1,)
    # these hyperparameters are for training on ~100M tokens (e.g. WikiText-103)
    # for training on smaller datasets, more dropout is necessary
    encoder_dps: Sequence[float] = tuple(default_dropout['classifier'][:-1])
    classifier_dps: Sequence[float] = (0.4, 0.1)
    lin_ftrs: Sequence[int] = (50,)
    drop_mult: float = 1.  # main switch to proportionally rescale dps
    clip: float = 0.12
    true_wd: bool = True

    # alpha and beta - defaults like in fastai/text/learner.py:RNNLearner()
    rnn_alpha: float = 2  # activation regularization (AR)
    rnn_beta: float = 1  # temporal activation regularization (TAR)

    cv_num_splits: int = 5
    cv_random_state: int = 17
    cv_fold_num: int = 0

    calc_test_score: bool = False
    load_best_validation_score: bool = True  # if True, version with the best val score is loaded, used for test
    # scoring and saved as the final model. Otherwise the last version is used
    train_set_fraction: float = 1.
    subsample_id: int = 1  # if train_set_fraction < 1, this selects which part goes into CV split. 0 -> old behaviour

    metrics: Sequence[str] = ('accuracy',)
    callbacks: Sequence[str] = ()

    @classmethod
    def factory(cls, name: str, params: Dict) -> 'ExperimentCls':
        return cls.subclass_registry[name](**params)

    @abstractmethod
    def get_dfs(self, fold_num: int) -> [DataFrame, DataFrame, DataFrame]:
        pass

    @property
    def cache_dir(self):
        return os.path.join(self.dataset_path, 'models', 'cls_cache')

    @property
    def vocab_path(self) -> str:
        return os.path.join(self.dataset_path, self.encoder_subdir, 'itos.pkl')

    @property
    def fwd_enc_path(self) -> str:
        return os.path.join(self.dataset_path, self.encoder_subdir, 'fwd_enc')

    @property
    def bwd_enc_path(self) -> str:
        return os.path.join(self.dataset_path, self.encoder_subdir, 'bwd_enc')

    @property
    def spm_vocab_path(self) -> str:
        return os.path.join(self.dataset_path, self.encoder_subdir, sentencepiece_tok.vocab_filename)

    @property
    def spm_model_path(self) -> str:
        return os.path.join(self.dataset_path, self.encoder_subdir, sentencepiece_tok.model_filename)

    @property
    def is_sentencepiece(self) -> bool:
        return all([os.path.exists(f) for f in [self.spm_model_path, self.spm_vocab_path]])

    def get_data_bunch(self, trn_df, val_df, tst_df) -> DataBunch:
        if self.is_sentencepiece:
            tokenizer, vocab = sentencepiece_tok.load_tokenizer(os.path.join(self.dataset_path, self.encoder_subdir))
        else:
            tokenizer = Tokenizer(tok_func=MosesTokenizerFunc, lang=self.lang, pre_rules=self.pre_rules,
                                  post_rules=self.post_rules)
            with open(self.vocab_path, 'rb') as file:
                vocab = Vocab(pickle.load(file))

        print(f"Running tokenization...")
        data_cls = TextClasDataBunch.from_df(path=self.cache_dir, train_df=trn_df, valid_df=val_df,
                                             test_df=tst_df, max_vocab=self.max_vocab, bs=self.bs,
                                             backwards=self.backwards, text_cols=self.text_cols,
                                             label_cols=self.label_col, mark_fields=self.mark_fields,
                                             tokenizer=tokenizer, vocab=vocab)
        print('Size of vocabulary:', len(data_cls.vocab.itos))
        print('First 20 words in vocab:', data_cls.vocab.itos[:20])
        return data_cls

    def get_learner(self, data_bunch: DataBunch,
                    agg_model: sequence_aggregations.Aggregation,
                    metrics: Optional[MetricFuncList] = None,
                    callbacks: Optional[List[Callback]] = None) -> 'RNNLearner':
        encoder_dps = [x * self.drop_mult for x in self.encoder_dps]
        num_classes = data_bunch.c
        vocab_size = len(data_bunch.vocab.itos)
        lin_ftrs = list(self.lin_ftrs) + [num_classes]
        rnn_enc = MultiBatchRNNCore(self.bptt, self.max_len, vocab_size, self.emb_sz, self.nh, self.nl,
                                    pad_token=PAD_TOKEN_ID, input_p=encoder_dps[0], weight_p=encoder_dps[1],
                                    embed_p=encoder_dps[2], hidden_p=encoder_dps[3])
        classifier = classifiers.SequenceAggregatingClassifier(agg_model, lin_ftrs, self.classifier_dps,
                                                               self.rnn_output_layers)
        model = SequentialRNN(rnn_enc, classifier)
        learn = RNNLearner(data_bunch, model, self.bptt, split_func=rnn_classifier_split, true_wd=self.true_wd,
                           metrics=metrics)
        learn.callback_fns += [StatsRecorder]
        if self.load_best_validation_score:
            learn.callback_fns += [partial(SaveModelCallback, every='improvement', name=CLS_BEST_FILE)]
        if callbacks:
            learn.callback_fns += callbacks
        return learn

    def get_bidir_learner(self, data_bunch: DataBunch,
                          agg_model: sequence_aggregations.Aggregation,
                          metrics: Optional[MetricFuncList] = None,
                          callbacks: Optional[List[Callback]] = None) -> 'RNNLearner':
        assert not self.backwards
        encoder_dps = [x * self.drop_mult for x in self.encoder_dps]
        num_classes = data_bunch.c
        vocab_size = len(data_bunch.vocab.itos)
        lin_ftrs = list(self.lin_ftrs) + [num_classes]
        enc_parts = [MultiBatchRNNCore(self.bptt, self.max_len, vocab_size, self.emb_sz, self.nh, self.nl,
                                       pad_token=PAD_TOKEN_ID, input_p=encoder_dps[0], weight_p=encoder_dps[1],
                                       embed_p=encoder_dps[2], hidden_p=encoder_dps[3]) for _ in range(2)]
        enc = classifiers.BidirEncoder(*enc_parts)
        classifier = classifiers.SequenceAggregatingClassifier(agg_model, lin_ftrs, self.classifier_dps,
                                                               self.rnn_output_layers)
        model = SequentialRNN(enc, classifier)
        learn = RNNLearner(data_bunch, model, self.bptt, split_func=classifiers.bidir_rnn_classifier_split,
                           true_wd=self.true_wd, metrics=metrics)
        learn.callback_fns += [StatsRecorder]
        if self.load_best_validation_score:
            learn.callback_fns += [partial(SaveModelCallback, every='improvement', name=CLS_BEST_FILE)]
        if callbacks:
            learn.callback_fns += callbacks
        return learn

    def run(self) -> Tuple[Dict, 'RNNLearner']:
        trn_df, val_df, tst_df = self.get_dfs(self.cv_fold_num)
        if self.train_set_fraction < 1:
            if self.subsample_id == 0:
                num_samples = int(len(trn_df) * self.train_set_fraction)
                trn_df = trn_df.iloc[:num_samples]
                val_df = val_df.iloc[:int(num_samples/3)]
            else:
                trn_df = trn_df.sample(frac=self.train_set_fraction, random_state=self.subsample_id)
                val_df = val_df.sample(frac=self.train_set_fraction, random_state=self.subsample_id)

        data_cls = self.get_data_bunch(trn_df, val_df, tst_df)

        agg_inp_size = 0
        for lnum in self.rnn_output_layers:
            if lnum == -1 or lnum == (self.nl-1):
                agg_inp_size += self.emb_sz
            else:
                agg_inp_size += self.nh
        if self.bidir:
            agg_inp_size *= 2

        agg_params = dict(dv=agg_inp_size, **self.aggregation_params)
        agg = sequence_aggregations.Aggregation.factory(self.aggregation_class, agg_params)
        metrics = [metrics_registry[m] for m in self.metrics]
        cbs = [callbacks_registry[c] for c in self.callbacks]
        if self.bidir:
            learn = self.get_bidir_learner(data_cls, agg, metrics, cbs)
            learn.model[0].enc1.load_state_dict(torch.load(self.fwd_enc_path + '.pth'))
            learn.model[0].enc2.load_state_dict(torch.load(self.bwd_enc_path + '.pth'))
        else:
            learn = self.get_learner(data_cls, agg, metrics, cbs)
            if self.backwards:
                learn.load_encoder(self.bwd_enc_path)
            else:
                learn.load_encoder(self.fwd_enc_path)

        results = {
            'phase_stats': [],
            'train_losses': []
        }

        for phase_params in self.training_phases:
            phase = Fit1CycleParams(**phase_params)
            learn.freeze_to(phase.freeze_to)
            learn.fit_one_cycle(**phase.to_dict())
            results['phase_stats'].append(copy(learn.stats_recorder.stats))
            results['train_losses'].append([l.item() for l in learn.recorder.losses])
        if self.calc_test_score:
            data_cls.train_dl = None
            data_cls.valid_dl = None
            gc.collect()
            torch.cuda.empty_cache()
            preds = learn.get_preds(DatasetType.Test, ordered=True)
            test_score = accuracy(preds[0], Tensor(tst_df[self.label_col]).long()).item()
            results['test_score'] = test_score
        results['best_val_acc'] = max([ep['accuracy'] for phase in results['phase_stats'] for ep in phase])
        return results, learn


@dataclass
class FactChecking(ExperimentCls):
    drop_mult = 1.
    mark_fields: bool = True
    pre_rules: Collection[Callable[[str], str]] = tuple([remove_urls] + defaults.text_pre_rules)
    text_cols: IntsOrStrs = ('category', 'subject', 'body')
    label_col: IntsOrStrs = 'fact_label'

    def get_dfs(self, fold_num: int = 0):
        trn_df_full = pd.read_csv(os.path.join(self.dataset_path, 'train.csv'))
        tst_df = pd.read_csv(os.path.join(self.dataset_path, 'test.csv'))

        kf = KFold(self.cv_num_splits, True, random_state=self.cv_random_state)
        split = list(kf.split(trn_df_full))[fold_num]
        trn_df = trn_df_full.iloc[split[0]]
        val_df = trn_df_full.iloc[split[1]]
        return trn_df, val_df, tst_df


@dataclass
class Imdb(ExperimentCls):
    cv_num_splits = 10
    drop_mult = 0.5

    def get_dfs(self, fold_num: int = 0):
        trn_df_full = pd.read_csv(os.path.join(self.dataset_path, 'train.csv'), header=None)
        tst_df = pd.read_csv(os.path.join(self.dataset_path, 'test.csv'), header=None)
        kf = KFold(self.cv_num_splits, True, random_state=self.cv_random_state)
        split = list(kf.split(trn_df_full))[fold_num]
        trn_df = trn_df_full.iloc[split[0]]
        val_df = trn_df_full.iloc[split[1]]
        return trn_df, val_df, tst_df


@dataclass
class Poleval1(ExperimentCls):

    def get_dfs(self, fold_num: int=0):
        # TODO: create separate validation and test
        df_full = pd.read_csv(os.path.join(self.dataset_path, 'task1_train.csv'), header=None)

        kf = KFold(self.cv_num_splits, True, random_state=self.cv_random_state)
        split = list(kf.split(df_full))[fold_num]
        trn_df = df_full.iloc[split[0]]
        val_df = df_full.iloc[split[1]]  # FIXME
        tst_df = pd.read_csv(os.path.join(self.dataset_path, 'task1_test.csv'), header=None)
        return trn_df, val_df, tst_df
