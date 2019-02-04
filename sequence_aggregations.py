from fastai import *
from fastai.text import *
import torch
import abc


class Aggregation(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def output_dir(self) -> int:
        """Should return the dimension of the returned hidden state (per-sample dimension)"""
        pass


class Baseline(nn.Module):
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


class SimpleAttention(nn.Module):
    def __init__(self, dv):
        super().__init__()
        self.dv = dv
        self.att_weights = nn.Parameter(Tensor(dv, 1))  # 2d for initializer to work
        nn.init.xavier_uniform_(self.att_weights.data)
        self.scale_fac = np.power(dv, 0.5)

    def forward(self, inp):
        bs = inp.shape[0]
        weights = torch.bmm(inp, self.att_weights.expand(bs, -1, -1)).squeeze()  # would repeat () be faster?
        weights /= self.scale_fac
        # attns = weights.relu_().softmax(1)  # not using softmax at all (but with relu) seems to be a good idea?
        attns = weights.softmax(1)
        weighted = inp * attns.unsqueeze(-1).expand_as(inp)  # bs x seq_len x dv
        result = weighted.sum(1)
        return result

    @property
    def output_dim(self) -> int:
        return self.dv


class NHeadDotProductAttention(nn.Module):
    def __init__(self, n_heads, dv):
        super().__init__()
        self.n_heads = n_heads
        self.dv = dv
        self.att_weights = nn.Parameter(Tensor(dv, n_heads))  # 2d for initializer to work
        nn.init.xavier_uniform_(self.att_weights.data)
        self.scale_fac = np.power(dv, 0.5)

    def forward(self, inp):
        bs = inp.shape[0]
        weights = torch.bmm(inp, self.att_weights.expand(bs, -1, -1)).squeeze()  # would repeat () be faster?
        weights /= self.scale_fac  # improves results a little
        attns = weights.relu_().softmax(1)  # lack of RELU breaks stuff
        weighted = inp.unsqueeze(-1).expand(-1, -1, -1, self.n_heads) * \
                   attns.unsqueeze(-2).expand(-1, -1, self.dv, -1)  # bs x seq_len x dv x n_heads
        result = weighted.sum(1).view(bs, -1)
        return result

    @property
    def output_dim(self) -> int:
        return self.dv * self.n_heads
