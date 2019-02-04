from fastai import *
from fastai.text import *
from . import sequence_aggregations


class SequenceAggregatingClassifier(nn.Module):
    def __init__(self, agg_mod: sequence_aggregations.Aggregation, layers: Collection[int], drops: Collection[float]):
        super().__init__()
        self.attn = agg_mod
        layers = [self.attn.output_dim] + list(layers)
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        raw_outputs, outputs = input
        output = outputs[-1]
        bs, sl, _ = output.size()  # bs, sl, nh
        x = self.attn(output)
        x = self.layers(x)
        return x, raw_outputs, outputs

