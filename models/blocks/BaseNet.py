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
            x = x.reshape(self.b, -1, x.shape[-1])
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


class BaseNet(nn.Module, BaseObject):
    def __init__(self, modality, Encoder_fn, Classifier_fn, name, variation=False):
        super().__init__()
        from params import model_modality_map
        self.encoders = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        for m in model_modality_map[modality]:
            self.encoders[m] = Encoder_fn(m=m, num_encoder=1)
            self.classifiers[m] = Classifier_fn(out_channel=self.encoders[m].out_channel, num_layers=1)
        if modality == 'MM':
            self.mm_classifier = Classifier_fn(out_channel=self.encoders[m].out_channel, num_layers=len(self.encoders))
        self.name = name
        self.variation = variation

    def kl_loss(self, m):
        if m == 'MM':
            return sum([encoder.kl_loss() for encoder in self.encoders.values()])
        return self.encoders[m].kl_loss()

    def modality(self, modality, data_dict, train_encoder):
        if modality == 'MM':
            if train_encoder:
                self.unfreeze([self.encoders, self.mm_classifier])
            else:
                self.freeze(self.encoders)
            out = []
            for m, x in data_dict.items():
                out.append(self.encoders[m](x))
            return self.mm_classifier(*out)
        if train_encoder:
            self.unfreeze([self.encoders[modality], self.classifiers[modality]])
        else:
            self.freeze(self.encoders[modality])
        out = self.encoders[modality](data_dict[modality])
        return self.classifiers[modality](out)

    def forward(self, inp, modality, train_encoder=True):
        if self.variation:
            return self.modality(modality, inp, train_encoder), self.kl_loss(modality)
        return self.modality(modality, inp, train_encoder)

    def encode(self, inp, modality):
        with torch.no_grad():
            if modality != 'MM':
                return {modality: self.encoders[modality](inp[modality])}
            out = {}
            for m in inp.keys():
                out.update({m: self.encoders[m](inp[m])})
            return out

    def classify(self, inp):
        if len(inp) > 1:
            out = []
            for m, x in inp.items():
                out.append(self.encoders[m](x))
            out = self.mm_classifier(*out)
            return out
        m = list(inp.keys())[0]
        return self.classifiers[m](inp[m])

    def get_classifier(self, modality):
        if modality == 'MM':
            return self.mm_classifier
        return self.classifiers[modality]

    def __getitem__(self, modality):
        '''
        :param modality: 根据输入 modality 返回模型不同部分，作为 Optimizer 输入
        :return:
        '''
        if modality == 'US' or modality == 'FA' or modality == 'ICGA':
            return nn.ModuleList([self.encoders[modality], self.classifiers[modality]])
        if modality == 'encoders':
            return self.encoders
        else:
            return self


class CycleBaseNet(BaseNet):
    def __init__(self, modality, Encoder_fn, out_features, name, variation=False):
        super().__init__(modality, Encoder_fn, lambda **kwargs: None, name, variation)
        from params import model_modality_map
        from .layers import CycleBlock
        from .classifiers import MMClassifier
        self.cycle_blocks = nn.ModuleDict()
        del self.classifiers
        for m in model_modality_map[modality]:
            self.encoders[m] = Encoder_fn(m=m, num_encoder=1)
            self.cycle_blocks[m] = nn.ModuleList(
                [CycleBlock(self.encoders[m].out_channel, num_layers=1) for _ in range(2)])
        self.mm_classifier = MMClassifier(in_features=self.encoders[m].out_channel, out_features=out_features)

    def modality(self, modality, data_dict, train_encoder):
        out = {}
        if train_encoder:
            self.unfreeze([self.encoders, self.cycle_blocks, self.mm_classifier])
        else:
            self.freeze(self.encoders)
        if modality == 'MM':
            for m, x in data_dict.items():
                out.update({m: self.encoders[m](x)})
        else:
            out.update({modality: self.encoders[modality](data_dict[modality])})
        # Fusion Block
        for cycle_id in range(2):
            for m in out.keys():
                others = [out[o] for o in out.keys() if o != m]
                out[m] = self.cycle_blocks[m][cycle_id](out[m], others)
        return self.mm_classifier(*out.values())

    def __getitem__(self, modality):
        '''
        :param modality: 根据输入 modality 返回模型不同部分，作为 Optimizer 输入
        :return:
        '''
        if modality == 'US' or modality == 'FA' or modality == 'ICGA':
            return nn.ModuleList([self.encoders[modality], self.mm_classifier])
        if modality == 'encoders':
            return self.encoders
        else:
            return self


class Prognosis(nn.Module, BaseObject):
    def __init__(self, state_path, model, Classifier_fn, name, train_encoder=False):
        super().__init__()
        from params import model_modality_map
        from utils.logger import char_color
        print(char_color(f'{name} Load model from {state_path}'))
        self.train_encoder = train_encoder
        model.load_state_dict(torch.load(state_path, map_location=torch.device('cpu'))['state_dict'])
        self.model = model
        self.encoders: nn.ModuleDict = model['encoders']
        self.classifiers = nn.ModuleDict()
        for m in model_modality_map['MM']:
            self.classifiers[m] = Classifier_fn(out_channel=self.encoders[m].out_channel, num_layers=1)
        self.mm_classifier = Classifier_fn(out_channel=self.encoders[m].out_channel, num_layers=len(self.encoders))
        self.name = name

    def infer_pathology(self, inp, modality):
        with torch.no_grad():
            return self.model(inp, modality)

    def modality(self, m, x):
        try:
            for i in self.classifiers.keys():
                if m != i:
                    self.freeze(self.classifiers[i])
            self.unfreeze(self.classifiers[m])
        except:
            raise Exception('Freeze Error')
        if self.train_encoder:
            x = self.encoders[m](x)
            x = self.classifiers[m](x)
            return x
        with torch.no_grad():
            x = self.encoders[m](x)
        x = self.classifiers[m](x)
        return x

    def multi_modality(self, data_dict):
        try:
            self.unfreeze(self.mm_classifier)
        except:
            raise Exception('Freeze Error')
        out = []

        if self.train_encoder:
            for m, x in data_dict.items():
                out.append(self.encoders[m](x))
            out = self.mm_classifier(*out)
            return out

        with torch.no_grad():
            for m, x in data_dict.items():
                out.append(self.encoders[m](x))
        out = self.mm_classifier(*out)
        return out

    def forward(self, inp, modality):
        if modality != 'MM':
            return self.modality(modality, inp[modality])
        return self.multi_modality(inp)

    def __getitem__(self, modality):
        if modality == 'US' or modality == 'FA' or modality == 'ICGA':
            if self.train_encoder:
                return nn.ModuleList([self.encoders[modality], self.classifiers[modality]])
            return self.classifiers[modality]
        if modality == 'MM':
            if self.train_encoder:
                return nn.ModuleList([self.encoders, self.mm_classifier])
            return self.mm_classifier
        raise Exception('Prognosis Error')

    def __setitem__(self, key, value):
        if key == 'US' or key == 'FA' or key == 'ICGA':
            self.classifiers[key] = value
        if key == 'encoders':
            self.encoders = value
        if key == 'MM':
            self.mm_classifier = value


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

    @torch.no_grad()
    def encode(self, modality, image):
        if modality != 'MM':
            return self.encoder({modality: image})[modality]
        else:
            features = self.encoder(image)
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
        if modality == 'classifier':
            return self.classifier
        if modality == 'encoders':
            return self.encoder
        else:
            return nn.ModuleList([self.encoder[modality], self.classifier[modality]])
