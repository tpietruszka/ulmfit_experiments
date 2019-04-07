from abc import ABCMeta, abstractmethod
from fastai.torch_core import *
from fastai.basic_train import Learner, LearnerCallback
from sklearn.metrics import roc_auc_score, average_precision_score


class StatsRecorder(LearnerCallback):
    """A `LearnerCallback` that saves history of metrics into a list of dicts. Based on fastai CSVLogger"""

    def __init__(self, learn: Learner):
        super().__init__(learn)
        self.stats = []

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        last_metrics = ifnone(last_metrics, [])
        ep_stats = {name: str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                    for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)}
        self.stats.append(ep_stats)


class RegisteredAbstractMeta(ABCMeta):
    """
    A class created by this metaclass and with `is_registry=True` will have a mapping of (name -> class) to all its
    subclasses in the `subclass_registry` class variable. Allows marking methods as abstract, as ABCMeta.

    Example:
    >>> class A(metaclass=RegisteredAbstractMeta, is_registry=True):
    ...     pass
    >>> class B(A):
    ...     def greet(self):
    ...         print('B-greet')
    >>> b_instance = A.subclass_registry['B']()
    >>> b_instance.greet()
    B-greet
    """

    def __new__(mcs, name, bases, class_dct, **kwargs):
        x = super().__new__(mcs, name, bases, class_dct)
        if kwargs.get('is_registry', False):
            x.subclass_registry = {}
        else:
            x.subclass_registry[name] = x
        return x


class MetricOnValidation(LearnerCallback):
    """modified from https://forums.fast.ai/t/using-auc-as-metric-in-fastai/38917/7"""

    def __init__(self, learn):
        super().__init__(learn)
        self.output, self.target = [], []

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names([self.name])

    def on_epoch_begin(self, **kwargs):
        self.output, self.target = [], []

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            self.output.append(last_output)
            self.target.append(last_target)

    def on_epoch_end(self, last_target, last_output, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            preds = F.softmax(output, dim=1)
            metric = self.metric_func(preds, target)
            self.learn.recorder.add_metrics([metric])

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def metric_func(self, input: Tensor, target: Tensor) -> float:
        pass


class AUROC(MetricOnValidation):
    _order = -20  # has to be unique and <0 to run before the loggers

    @property
    def name(self):
        return 'AUROC'

    def metric_func(self, input, target):
        input, target = input.cpu().numpy()[:, 1], target.cpu().numpy()
        return roc_auc_score(target, input)


class AveragePrecisionScore(MetricOnValidation):
    _order = -19

    @property
    def name(self):
        return 'avg_precision'

    def metric_func(self, input, target):
        input, target = input.cpu().numpy()[:, 1], target.cpu().numpy()
        return average_precision_score(target, input)
