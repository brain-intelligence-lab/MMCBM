import random

import numpy as np
import torch
import torch.nn as nn
from .layers import MergeLinearAttention, LinearSelfAttention, SelfAttentionPooling
from collections import OrderedDict


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, attn=True):
        super(Classifier, self).__init__()
        self.attn = None
        self.multi = num_layers > 1
        if num_layers > 1:
            if attn:
                self.attn = MergeLinearAttention(in_features=in_features, num_layers=num_layers)
            else:
                in_features = in_features * num_layers

        # self.classifier = nn.Sequential(OrderedDict([
        #     ('mlp', nn.Linear(in_features, 128)),
        #     ('relu', nn.ReLU(inplace=True)),
        #     ('out', nn.Linear(128, out_features)),
        # ]))
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, *x):
        if self.multi and self.attn is not None:
            x = self.attn(*x)
        else:
            x = torch.cat(x, dim=-1)
        return self.classifier(x)


class PrognosisClassifier(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, attn=True):
        super(PrognosisClassifier, self).__init__()
        self.attn = None
        self.multi = num_layers > 1
        if num_layers > 1:
            if attn:
                self.attn = MergeLinearAttention(in_features=in_features, num_layers=num_layers)
            else:
                in_features = in_features * num_layers

        self.classifier = nn.Sequential(OrderedDict([
            ('mlp', nn.Linear(in_features, 128)),
            ('relu', nn.LeakyReLU(.2, inplace=True)),
            ('out', nn.Linear(128, out_features)),
        ]))

    def forward(self, *x):
        if self.multi and self.attn is not None:
            x = self.attn(*x)
        else:
            x = torch.cat(x, dim=-1)
        return self.classifier(x)


class AvgMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = nn.Conv1d(in_channels=2,
                                     out_channels=1,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False)

    def forward(self, x):
        avg = x.mean(dim=1)
        max = x.max(dim=1)[0]

        return self.down_sample(torch.stack([avg, max], dim=1)).squeeze(1)


class MMClassifier(nn.Module):
    def __init__(self, in_features, out_features, reduction=4, mask_prob=0):
        super().__init__()
        # self.pool = TimeRNNAttentionPooling(input_size=in_features, num_layers=2, pure_out=True)
        self.pool = AvgMaxPool()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // reduction),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(in_features=in_features // reduction, out_features=out_features)
        )
        self.R = np.random.RandomState()
        self.mask_prob = mask_prob

    def do(self):
        return self.R.rand() < self.mask_prob

    def forward(self, *x):
        x = list(x)
        if self.do():
            x.pop(np.random.randint(0, len(x)))
        x = torch.stack(x, dim=1)
        x = self.pool(x)
        return self.classifier(x)


class SimpleClassifier(nn.Module):
    def __init__(self, in_features, out_features, reduction=16):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.Dropout(.2),
            nn.Linear(in_features // reduction, out_features)
        )

    def forward(self, x):
        return self.classifier(x)


class RNNClassifier(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, reduction=8, mask_prob=0.5):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_features, hidden_size=in_features // 2,
                           num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attn = LinearSelfAttention(in_features=in_features, bias=False)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.classifier = SimpleClassifier(in_features=in_features,
                                           out_features=out_features,
                                           reduction=reduction)
        self.R = np.random.RandomState()
        self.mask_prob = mask_prob

    def __getitem__(self, item):
        return self

    def do(self):
        return self.R.rand() < self.mask_prob

    def random_mask(self, x: list):
        if isinstance(x, list) and self.do and len(x) > 1:
            x.pop(random.randint(0, len(x) - 1))
        return x

    def random_shuffle(self, x: list):
        if isinstance(x, list) and self.do():
            np.random.shuffle(x)
        return x

    def fusion(self, x):
        x = list(x.values()) if isinstance(x, dict) else x
        x = torch.concat(x, dim=1)
        x = self.rnn(x)[0][:, -1]
        return x

    def forward(self, x, modality=None):
        x = self.fusion(x)
        x = self.attn(x)
        x = self.relu(x)
        return self.classifier(x)


class AttnPoolClassifier(nn.Module):
    def __init__(self, in_features, out_features, reduction=8, mask_prob=0.5):
        super().__init__()
        self.pool = SelfAttentionPooling(input_dim=in_features)
        self.attn = LinearSelfAttention(in_features=in_features, bias=False)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.classifier = SimpleClassifier(in_features=in_features,
                                           out_features=out_features,
                                           reduction=reduction)
        self.mask_prob = mask_prob

    def __getitem__(self, item):
        return self

    def do(self):
        return self.R.rand() < self.mask_prob

    def fusion(self, x):
        x = list(x.values()) if isinstance(x, dict) else x
        x = torch.concat(x, dim=1)
        if x.shape[1] > 1:
            x = self.pool(x)
        else:
            x = x.squeeze(1)
        return x

    def forward(self, x, modality=None):
        x = self.fusion(x)
        # x = self.pool(x)
        x = self.attn(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class MaxPoolClassifier(nn.Module):
    def __init__(self, in_features, out_features, reduction=8, mask_prob=0.5):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.attn = LinearSelfAttention(in_features=in_features, bias=False)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.classifier = SimpleClassifier(in_features=in_features,
                                           out_features=out_features,
                                           reduction=reduction)
        self.mask_prob = mask_prob

    def __getitem__(self, item):
        return self

    def fusion(self, x):
        x = list(x.values()) if isinstance(x, dict) else x
        x = torch.concat(x, dim=1)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(2)
        return x

    def forward(self, x, modality=None):
        x = self.fusion(x)
        x = self.attn(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class MMDictClassifier(nn.Module):
    def __init__(self, in_features, out_features, num_layers, reduction=8, mask_prob=0.5):
        super().__init__()
        if num_layers > 0:
            self.rnn = nn.GRU(input_size=in_features, hidden_size=in_features // 2,
                              num_layers=num_layers, bidirectional=True, batch_first=True)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.classifiers = nn.ModuleDict({
            modality: SimpleClassifier(
                in_features=in_features,
                out_features=out_features,
                reduction=reduction
            ) for modality in ['FA', 'ICGA', 'US', 'MM']
        })
        self.R = np.random.RandomState()
        self.mask_prob = mask_prob

    def __getitem__(self, item):
        modules = nn.ModuleList()
        if hasattr(self, 'rnn'):
            modules.append(self.rnn)
        modules.append(self.classifiers[item])
        return modules

    def do(self):
        return self.R.rand() < self.mask_prob

    def random_shuffle(self, x: list):
        if self.do():
            np.random.shuffle(x)
        return x

    def forward(self, x: dict, modality=None):
        if len(x) == 1:
            modality = list(x.keys())[0]
        else:
            modality = 'MM'
        x = torch.concat(list(x.values()), dim=1)
        if hasattr(self, 'rnn'):
            x = self.rnn(x)[0][:, -1]
        x = self.relu(x)
        return self.classifiers[modality](x)


class MMFusionClassifier(nn.Module):
    def __init__(self, in_features, out_features, num_layers, reduction=8, mask_prob=0):
        super().__init__()
        self.classifiers = nn.ModuleDict({
            modality: RNNClassifier(in_features=in_features, out_features=out_features,
                                    num_layers=num_layers if modality != 'US' else 0,
                                    reduction=reduction)
            for modality in ['FA', 'ICGA', 'US', 'MM']
        })
        self.R = np.random.RandomState()
        self.mask_prob = mask_prob

    def __getitem__(self, item):
        return self.classifiers[item]

    def do(self):
        return self.R.rand() < self.mask_prob

    def random_shuffle(self, x: list):
        if self.do():
            np.random.shuffle(x)
        return x

    def forward(self, x: dict, modality=None):
        if len(x) > 1:
            return self.classifiers['MM'](x)
        modality = list(x.keys())[0]
        return self.classifiers[modality](x)
