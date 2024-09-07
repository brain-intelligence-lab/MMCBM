# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import torch
import torch.nn as nn


class BaseObject:
    def freeze(self, net):
        if not isinstance(net, (list, tuple)):
            net = [net]
        for n in net:
            for param in n.parameters():
                param.requires_grad = False

    def unfreeze(self, net):
        if not isinstance(net, (list, tuple)):
            net = [net]
        for n in net:
            for param in n.parameters():
                param.requires_grad = True

    def modality(self, m, x):
        pass

    def multi_modality(self, data_dict):
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass


class SingleBaseEncoder(nn.Module):
    def __init__(self, out_channel, spatial_dims=2, enc_no_grad=False):
        super().__init__()
        self.out_channel = out_channel
        self.spatial_dims = spatial_dims
        self.encoder = None
        self.enc_no_grad = enc_no_grad

    def to_B(self, x):
        if self.spatial_dims == 2:
            self.b = x.shape[0]
            x = x.reshape(-1, *x.shape[2:])
        return x

    def to_T(self, x):
        if self.spatial_dims == 2:
            x = x.reshape(self.b, -1, *x.shape[1:])
        return x

    def encode(self, x, m):
        x = self.encoder(x)
        return x

    def forward(self, x: dict):
        out = {}
        for m, v in x.items():
            v = self.to_B(v)
            if self.enc_no_grad and len(x) > 1:
                with torch.no_grad():
                    v = self.encode(v, m)
            else:
                v = self.encode(v, m)
            v = self.to_T(v)
            out[m] = v
        return out

    def __getitem__(self, item):
        return self


class MMBaseEncoder(SingleBaseEncoder):
    def encode(self, x, m):
        return self.encoder[m](x)

    def __getitem__(self, item):
        return self.encoder[item]


class BaseEncoder(nn.Module):
    def __init__(self, out_channel, concat=False, bidirectional=False, spatial_dims=2):
        super().__init__()
        self.out_channel = out_channel
        self.spatial_dims = spatial_dims
        self.encoders = nn.ModuleList()
        self.rnns = nn.ModuleList()
        if concat:
            import pdb
            pdb.set_trace()
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=concat, out_channels=12, kernel_size=1),
                nn.BatchNorm1d(12),
                nn.ReLU(),
                nn.Conv1d(in_channels=12, out_channels=1, kernel_size=1)
            )
        if bidirectional:
            self.rnns.append(nn.LSTM(input_size=out_channel, hidden_size=out_channel // 2, bidirectional=True))

    def to_B(self, x):
        if self.spatial_dims == 2:
            self.time_step = x.shape[1]
            x = x.reshape(-1, *x.shape[2:])
        return x

    def to_T(self, x):
        if self.spatial_dims == 2:
            x = x.reshape(-1, self.time_step, x.shape[-1])
        return x

    def to_rnn(self, x, idx):
        if self.spatial_dims == 2:
            if len(self.rnns) > 0:
                x = self.rnns[idx](x)[0][:, -1, :]

            elif self.time_step > 1:
                x = self.downsample(x).reshape(-1, x.shape[-1])
            else:
                x = x.reshape(-1, x.shape[-1])
        return x

    def forward(self, *x):
        out = []
        for i, data in enumerate(x):
            x_i = self.to_B(data)
            x_i = self.encoders[i](x_i)
            x_i = self.to_T(x_i)

            x_i = self.to_rnn(x_i, i)
            out.append(x_i)
        out = self.attn(*out)
        return out


class VariationEncoder(BaseEncoder):
    def __init__(self, istrain=True):
        super().__init__(0, concat=False, bidirectional=False)
        self.variations = nn.ModuleList()
        self.vars = []
        self.istrain = istrain

    def reparameters(self, x):
        mu, logvar = torch.chunk(x, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        if self.istrain:
            std *= torch.randn_like(std)
        return mu + std

    def forward(self, *x):
        self.vars.clear()
        for i, data in enumerate(x):
            x_i = self.to_B(data)
            x_i = self.encoders[i](x_i)
            x_i = self.to_rnn(x_i, i)
            x_i = self.variations[i](x_i)
            self.vars.append(x_i)
            z = self.reparameters(x_i)
            return z

    def kl_loss(self):
        loss = 0
        for var in self.vars:
            mu, logvar = torch.chunk(var, 2, dim=-1)
            loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return loss.mean(dim=0)


class SingleBaseNet(nn.Module, BaseObject):
    """
    Single FusionEncoder for all modalities
    ignore the modality difference in vision information encoding
    """

    def __init__(self, Encoder_fn, Classifier_fn, name):
        super().__init__()
        self.encoder = Encoder_fn()
        self.classifier = Classifier_fn(out_channel=self.encoder.out_channel)
        self.name = name

    def modality(self, modality, data_dict):
        if modality != 'MM':
            data_dict = {modality: data_dict[modality]}
        return self.encoder(data_dict)

    def forward(self, inp, modality, train_encoder=True):
        features = self.modality(modality, inp)
        return self.classifier(features, modality)

    def encode_image(self, modality, image, keep_dict=False):
        if not isinstance(image, dict):
            image = {modality: image}
        if modality != 'MM' and modality in image.keys():
            return self.encoder({modality: image[modality]})[modality]
        else:
            features = self.encoder(image)
            if keep_dict:
                return features
            return self.classifier.fusion(features)

    def classify(self, inp):
        out = []
        for m, x in inp.items():
            out.append(self.encoder(x))
        return self.classifier(*out)

    def get_classifier(self):
        return self.classifier

    def __getitem__(self, modality):
        '''
        :param modality: 根据输入 modality 返回模型不同部分，作为 Optimizer 输入
        :return:
        '''
        if modality == 'classifiers':
            return self.classifier
        if modality == 'encoders':
            return self.encoder
        else:
            return nn.ModuleDict({
                'encoder': self.encoder[modality],
                'classifier': self.classifier[modality]
            })
