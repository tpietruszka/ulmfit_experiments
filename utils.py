from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import Learner, LearnerCallback

__all__ = ['StatsRecorder']


class StatsRecorder(LearnerCallback):
    """A `LearnerCallback` that saves history of metrics into a list of dicts"""

    def __init__(self, learn: Learner):
        super().__init__(learn)
        self.stats = []

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        last_metrics = ifnone(last_metrics, [])
        ep_stats = {name: str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                    for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)}
        self.stats.append(ep_stats)
