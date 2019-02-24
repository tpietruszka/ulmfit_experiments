from abc import ABCMeta
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import Learner, LearnerCallback

__all__ = ['StatsRecorder', 'RegisteredAbstractMeta']


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
