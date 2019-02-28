from fastai import *
from fastai.text import *
from . import sequence_aggregations


class SequenceAggregatingClassifier(nn.Module):
    def __init__(self, agg_mod: sequence_aggregations.Aggregation, layers: Collection[int], drops: Collection[float],
                 output_layers: List[int]):
        super().__init__()
        self.attn = agg_mod
        layers = [self.attn.output_dim] + list(layers)
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)
        self.output_layers = output_layers

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        raw_outputs, outputs = input
        if len(self.output_layers) == 1:
            output = outputs[self.output_layers[0]]
        else:
            output = torch.cat([outputs[x] for x in self.output_layers], 2)
        bs, sl, _ = output.size()  # bs, sl, nh
        x = self.attn(output)
        x = self.layers(x)
        return x, raw_outputs, outputs


class BidirEncoder(nn.Module):
    def __init__(self, enc1, enc2):
        super().__init__()
        self.enc1 = enc1
        self.enc2 = enc2

    def forward(self, input: LongTensor) -> Tuple[LongTensor, LongTensor]:
        e1 = self.enc1(input)
        e2 = self.enc2(input.flip(1))

        raw = [torch.cat([x1, x2.flip(1)], 2) for x1, x2 in zip(e1[0], e2[0])]
        out = [torch.cat([x1, x2.flip(1)], 2) for x1, x2 in zip(e1[1], e2[1])]
        return raw, out

    def reset(self):
        self.enc1.reset()
        self.enc2.reset()


def bidir_rnn_classifier_split(model: nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    enc_groups = []
    for enc in [model[0].enc1, model[0].enc2]:
        g1 = [[enc.encoder, enc.encoder_dp]]
        g1 += [[rnn, dp] for rnn, dp in zip(enc.rnns, enc.hidden_dps)]
        enc_groups.append(g1)
    groups = [sum(gr, []) for gr in zip(*enc_groups)]
    groups.append([model[1]])
    return groups
