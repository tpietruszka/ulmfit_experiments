from fastai import *
from fastai.text import *
import torch
import abc
from .utils import RegisteredAbstractMeta


class Aggregation(nn.Module, metaclass=RegisteredAbstractMeta, is_registry=True):
    @abc.abstractmethod
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def output_dim(self) -> int:
        """Should return the dimension of the returned hidden state (per-sample dimension)"""
        pass

    @classmethod
    def factory(cls, name: str, params: Dict) -> 'Aggregation':
        return cls.subclass_registry[name](**params)


class Baseline(Aggregation):
    "Create a linear classifier with pooling."

    def __init__(self, dv):
        super().__init__()
        self.dv = dv

    def pool(self, x: Tensor, bs: int, is_max: bool):
        "Pool the tensor along the seq_len dimension."
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.transpose(1, 2), (1,)).view(bs, -1)

    def forward(self, inp: Tensor) -> Tensor:
        bs, sl, _ = inp.size()
        avgpool = self.pool(inp, bs, False)
        mxpool = self.pool(inp, bs, True)
        return torch.cat([inp[:, -1], mxpool, avgpool], 1)

    @property
    def output_dim(self) -> int:
        return 3 * self.dv


class SimpleAttention(Aggregation):
    def __init__(self, dv):
        super().__init__()
        self.dv = dv
        self.att_weights = nn.Parameter(Tensor(dv, 1))  # 2d for initializer to work
        nn.init.xavier_uniform_(self.att_weights.data)
        self.scale_fac = np.power(dv, 0.5)
        self.att_scores = None

    def forward(self, inp):
        bs = inp.shape[0]
        weights = torch.bmm(inp, self.att_weights.expand(bs, -1, -1)).squeeze()  # would repeat () be faster?
        weights /= self.scale_fac
        # attns = weights.relu_().softmax(1)  # not using softmax at all (but with relu) seems to be a good idea?
        attns = weights.softmax(1)
        weighted = inp * attns.unsqueeze(-1).expand_as(inp)  # bs x seq_len x dv
        result = weighted.sum(1)
        self.att_scores = attns
        return result

    @property
    def output_dim(self) -> int:
        return self.dv


class SimpleDropConnectAttention(Aggregation):
    def __init__(self, dv, p_drop):
        super().__init__()
        self.att = SimpleAttention(dv)
        self.wrapped_att = WeightDropout(self.att, p_drop, ['att_weights'])

    def forward(self, *args, **kwargs):
        return self.wrapped_att(*args, **kwargs)

    @property
    def output_dim(self):
        return self.att.output_dim


class NHeadDotProductAttention(Aggregation):
    def __init__(self, n_heads, dv):
        super().__init__()
        self.n_heads = n_heads
        self.dv = dv
        self.att_weights = nn.Parameter(Tensor(dv, n_heads))  # 2d for initializer to work
        nn.init.xavier_uniform_(self.att_weights.data)
        self.scale_fac = np.power(dv, 0.5)

    def forward(self, inp):
        bs = inp.shape[0]
        weights = torch.bmm(inp, self.att_weights.expand(bs, -1, -1))  # would repeat () be faster?
        weights /= self.scale_fac  # improves results a little
        attns = weights.relu_().softmax(1)  # lack of RELU breaks stuff
        weighted = inp.unsqueeze(-1).expand(-1, -1, -1, self.n_heads) * \
                   attns.unsqueeze(-2).expand(bs, -1, self.dv, -1)  # bs x seq_len x dv x n_heads
        result = weighted.sum(1).view(bs, -1)
        return result

    @property
    def output_dim(self) -> int:
        return self.dv * self.n_heads


class MultiLayerPointwise(nn.Module):
    """Applies a point-wise neural network, with ReLU activations between layers and none at the end"""

    def __init__(self, dims: Sequence[int], dropouts: Union[Sequence[float], float] = 0., batchnorm: bool = True):
        super().__init__()
        acts = [nn.ReLU(inplace=True)] * (len(dims) - 2) + [None]
        layers = []
        if not isinstance(dropouts, Sequence):
            dropouts = [dropouts] * (len(dims) - 1)
        for din, dout, act, drop in zip(dims[:-1], dims[1:], acts, dropouts):
            layers += bn_drop_lin(din, dout, bn=batchnorm, p=drop, actn=act)
        self.layers = nn.Sequential(*layers)

    def forward(self, inp):
        bs, sl, dv = inp.shape
        x = inp.view(bs * sl, -1)
        x = self.layers(x)
        x = x.view(bs, sl, -1)
        return x


class BranchingAttentionAggregation(Aggregation):
    def __init__(self, dv: int, att_hid_layers: Sequence[int], att_dropouts: Sequence[float],
                 agg_dim: Optional[int] = None, add_last_el: bool = False):
        super().__init__()
        att_layers = [dv] + list(att_hid_layers) + [1]
        self.head = MultiLayerPointwise(att_layers, att_dropouts, batchnorm=False)
        self.agg_dim = agg_dim
        if agg_dim:
            self.agg = MultiLayerPointwise([dv, agg_dim], 0, batchnorm=False)
        self.add_last_el = add_last_el
        if add_last_el:
            self.last_el_weight = nn.Parameter(tensor(0.5))
        self.dv = dv
        self.last_weights = None

    @property
    def output_dim(self):
        return self.agg_dim or self.dv

    def forward(self, inp):
        weights = F.softmax(self.head(inp).squeeze(), dim=1)
        self.last_weights = weights
        if self.agg_dim:
            to_agg = self.agg(inp)
        else:
            to_agg = inp
        weighted = to_agg * weights.unsqueeze(-1).expand_as(to_agg)
        if self.add_last_el:
            res = weighted.sum(1) * (1-self.last_el_weight) + self.last_el_weight * to_agg[:, -1, :]
        else:
            res = weighted.sum(1)
        return res
