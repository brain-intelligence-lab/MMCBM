import torch
import torch.nn as nn
from monai.networks.layers import Pool, get_act_layer
from monai.networks.nets.densenet import _load_state_dict
from monai.utils import ensure_tuple_rep
from monai.networks.nets import DenseNet
from monai.networks.nets.swin_unetr import SwinTransformer
from collections import OrderedDict
from .BaseNet import BaseEncoder


class ReductionLinear(nn.Module):
    def __init__(self, in_features, out_features, reduction):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // reduction),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(in_features=in_features // reduction, out_features=out_features)
        )

    def forward(self, x):
        return self.mlp(x)


class AttentionRnnBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.wq = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.wk = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.wv = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

    def forward(self, x, hidden):
        """
        :param x: (N, T, H)
        :param hidden_t1: (N, 1, H)
        :return:
        """
        if hidden is None:
            hidden = torch.zeros((x.shape[0], x.shape[2])).to(x.device)
        hiddens = []
        for t in range(x.shape[1]):
            hiddens.append(hidden)
            query = self.wq(torch.cat([x[:, t, :], hidden], dim=1))
            key = self.wk(torch.cat([x[:, t, :], hidden], dim=1))
            value = self.wv(torch.cat([x[:, t, :], hidden], dim=1))
            w = torch.sigmoid(query * key)
            hidden = (1 - w) * hidden + w * torch.tanh(value)
        return torch.stack(hiddens, dim=1), hidden


class AttentionRnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.prob = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.attentions = nn.ModuleList([AttentionRnnBlock(hidden_size) for _ in range(num_layers)])

    def forward(self, x, hidden):
        x = self.prob(x)
        for attention in self.attentions:
            x, hidden = attention(x, hidden)
        return x, hidden


class TimeRNNAttentionPooling(nn.Module):
    """
    Implementation of TimeRNNAttentionPooling
    任意时间步的 Pooling
    """

    def __init__(self, input_size, num_layers=2, pure_out=False):
        super().__init__()
        # self.time_weight = AttentionRnn(input_size=input_size, hidden_size=1, num_layers=num_layers)
        # self.spatial_weight = AttentionRnn(input_size=input_size, hidden_size=input_size, num_layers=num_layers)
        self.time_weight = nn.GRU(input_size=input_size, hidden_size=1, num_layers=num_layers,
                                  batch_first=True, bidirectional=True)
        self.prob = nn.Linear(2, 1)
        self.pure_out = pure_out

    def forward(self, x):
        """
            input:
                batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
            attention_weight:
                time_weight : size (N, T, 1)
                spatial_weight : size (N, T, H)
            return:
                utter_rep: size (N, H)
        """

        ta = torch.softmax(self.prob(self.time_weight(x)[0]), dim=1)
        if self.pure_out:
            return (x * ta).sum(1)
        x = (x * ta).sum(1, keepdim=True)
        return x,


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    code from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.weight = ReductionLinear(input_dim, input_dim, reduction=8)

    def forward(self, x):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        attention_weight:
            att_w : size (N, T, 1)
        return:
            utter_rep: size (N, H)
        """

        # (N, T, H) -> (N, T) -> (N, T, 1)
        att_w = nn.functional.softmax(self.weight(x), dim=1)
        x = torch.sum(x * att_w, dim=1)
        return x


class SelfLinearAttentionPooling(SelfAttentionPooling):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.weight = nn.Linear(input_dim, input_dim, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x, skip=None):
        b, c, _, _ = x.size()
        out = x * self.ca(x)
        if skip is None:
            out = out * self.sa(out)
        else:
            out = skip * self.sa(out)
        return out


class ResidualCBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.residual_cba = CBAMBlock(channel, reduction, kernel_size)
        self.inp = CBAMBlock(channel, reduction, kernel_size)

    def forward(self, x, residual):
        residual = self.residual_cba(x, residual)
        x = self.inp(x)
        return x + residual


class LinearChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.shape[-1] > 1:
            max_result = self.maxpool(x)
            avg_result = self.avgpool(x)
            output = self.se(max_result) + self.se(avg_result)
        else:
            output = self.se(x)
        output = self.sigmoid(output)
        return output * x


class LinearSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output * x


class LinearCBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, residual=False):
        super().__init__()
        self.residual = residual
        self.ca = LinearChannelAttention(channel=channel, reduction=reduction)
        self.sa = LinearSpatialAttention()

    def forward(self, x):
        shape = x.shape
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        out = self.ca(x)
        out = self.sa(out)
        if self.residual:
            out = out + x
        return out.reshape(shape)


class ReductionLinearSelfAttention(nn.Module):
    def __init__(self, in_features, reduction):
        super().__init__()
        self.weight = ReductionLinear(in_features, in_features, reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.weight(x)) * x


class LinearSelfAttention(nn.Module):
    def __init__(self, in_features, bias=True):
        super().__init__()
        self.linear_q = nn.Linear(in_features, in_features, bias=bias)
        self.linear_k = nn.Linear(in_features, in_features, bias=bias)
        self.linear_v = nn.Linear(in_features, in_features, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        attention = self.sigmoid(q * k)
        return attention * v


class LinearDotSelfAttention(nn.Module):
    def __init__(self, in_features, bias=True):
        super().__init__()
        self.linear_q = nn.Linear(in_features, in_features, bias=bias)
        self.linear_k = nn.Linear(in_features, in_features, bias=bias)
        self.linear_v = nn.Linear(in_features, in_features, bias=bias)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        q = q.unsqueeze(-1)
        k = k.unsqueeze(-2)
        attention = torch.softmax(torch.matmul(q, k), dim=-1)
        return torch.matmul(attention, v.unsqueeze(-1)).squeeze(-1)


class MergeLinearAttention(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=None,
                 reduction=16,
                 residual=False,
                 num_layers=1,
                 ):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param bias: Whether to use the bias term.
        """
        super(MergeLinearAttention, self).__init__()
        out_features = out_features if out_features is not None else in_features
        self.input_fc = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_layers)]) \
            if in_features != out_features else None
        self.num_layers = num_layers
        self.lsas = nn.ModuleList([LinearSelfAttention(out_features) for _ in range(num_layers)])
        self.out = LinearCBAMBlock(out_features, reduction, residual) if num_layers > 1 else None
        self.down_sample = nn.Conv1d(in_channels=num_layers, out_channels=1, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, *x):
        out = []
        for i, data in enumerate(x):
            if self.input_fc is not None:
                data = self.input_fc[i](data)
            out.append(self.lsas[i](data))
        out = torch.stack(out, dim=-1)
        if self.num_layers > 1:
            out = self.out(out)
        out = self.down_sample(out.permute(0, 2, 1)).view(out.shape[0], -1)
        out = self.relu(out)
        return out


class CycleAttention(nn.Module):
    def __init__(self, in_features, dot=True):
        super().__init__()
        self.linear_q = nn.Linear(in_features, in_features, bias=False)
        self.linear_k = nn.Linear(in_features, in_features, bias=False)
        self.linear_v = nn.Linear(in_features, in_features, bias=False)
        self.dot = dot

    def forward(self, value, query=None, key=None):
        if query is None and key is None:
            return value
        if query is None:
            query = value
        if key is None:
            key = value

        value = self.linear_v(value)
        query = self.linear_q(query)
        key = self.linear_k(key)
        if self.dot:
            value = value.unsqueeze(-1)
            query = query.unsqueeze(-1)
            key = key.unsqueeze(-2)
            attention = torch.softmax(torch.einsum('b j i, b i k -> b j k', query, key), dim=-1)
            return torch.einsum('b j k, b k i -> b j i', attention, value).squeeze(-1)
        return torch.sigmoid(query * key) * value


class CycleBlock(nn.Module):
    def __init__(self, features=512, num_layers=1):
        super().__init__()
        self.cycle_model = nn.ModuleList(CycleAttention(in_features=features, dot=False) for _ in range(num_layers))
        # self.self_attn = LinearDotSelfAttention(in_features=features, bias=True)
        self.self_attn = LinearSelfAttention(in_features=features, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, main, others):
        for cycle_model in self.cycle_model:
            main = cycle_model(main, *others)
        main = self.self_attn(main)
        main = self.relu(main)
        return main


class MAModule(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(MAModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X M X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, M, C, height, width = x.size()
        qkv = x.view(m_batchsize, M, -1)

        energy = torch.bmm(qkv, qkv.permute(0, 2, 1))  # (B X M X M)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out = torch.bmm(attention, qkv)  # (B X M X M) * (B X M X N) = (B X M X N)
        out = out.view(m_batchsize, M, C, height, width).sum(1)
        return out


########################################## Merge Encoders ##########################################


class Dense121Net(DenseNet):
    def __init__(self,
                 spatial_dims,
                 in_channels: int,
                 init_features: int = 64,
                 growth_rate: int = 32,
                 block_config=(6, 12, 24, 16),
                 bn_size: int = 4,
                 act=("relu", {"inplace": True}),
                 norm="batch",
                 dropout_prob: float = 0.0,
                 pretrained=True):
        super(Dense121Net, self).__init__(spatial_dims=spatial_dims,
                                          in_channels=in_channels,
                                          out_channels=1,
                                          init_features=init_features,
                                          growth_rate=growth_rate,
                                          block_config=block_config,
                                          bn_size=bn_size,
                                          act=act,
                                          norm=norm,
                                          dropout_prob=dropout_prob)
        avg_pool_type = Pool[Pool.ADAPTIVEAVG, 2]
        self.out_channels = init_features
        for i, num_layers in enumerate(block_config):
            self.out_channels += num_layers * growth_rate
            if i != len(block_config) - 1:
                self.out_channels = self.out_channels // 2
        if pretrained:
            _load_state_dict(self, "densenet121", True)
        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                ]
            )
        )


class DenseNetEncoder(BaseEncoder):
    def __init__(self, spatial_dims, input_channels=3, num_layers=0, pretrained: bool = False, encoder_num=1,
                 ):
        super(DenseNetEncoder, self).__init__(out_channel=0, concat=False, spatial_dims=spatial_dims)

        for _ in range(encoder_num):
            self.encoders.append(Dense121Net(spatial_dims=spatial_dims,
                                             in_channels=input_channels,
                                             pretrained=pretrained))
            self.out_channel = self.encoders[0].out_channels
            if num_layers != 0 and spatial_dims == 2:
                self.rnns.append(TimeRNNAttentionPooling(
                    input_size=self.out_channel,
                    num_layers=num_layers)
                )
        self.attn = MergeLinearAttention(in_features=self.out_channel,
                                         out_features=self.out_channel,
                                         num_layers=encoder_num)


class SwinTransformerEncoder(SwinTransformer):
    def __init__(self,
                 spatial_dims: int = 2,
                 input_channels=3,
                 depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24),
                 feature_size: int = 24,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 dropout_path_rate: float = 0.0,
                 use_checkpoint: bool = False,
                 ):
        super().__init__(in_chans=input_channels,
                         embed_dim=feature_size,
                         window_size=ensure_tuple_rep(7, spatial_dims),
                         patch_size=ensure_tuple_rep(2, spatial_dims),
                         depths=depths,
                         num_heads=num_heads,
                         mlp_ratio=4.0,
                         qkv_bias=True,
                         drop_rate=drop_rate,
                         attn_drop_rate=attn_drop_rate,
                         drop_path_rate=dropout_path_rate,
                         norm_layer=nn.LayerNorm,
                         use_checkpoint=use_checkpoint,
                         spatial_dims=spatial_dims)
        self.out_channel = feature_size * 2 ** len(depths)
        self.avg = nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x1 = self.layers1[0](x0.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return torch.flatten(self.avg(x4_out), 1)


class SwinEncoder(BaseEncoder):
    def __init__(self,
                 spatial_dims: int = 2,
                 input_channels=3,
                 lstm=0,
                 concat=False,
                 encoder_num=1,
                 depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24),
                 feature_size: int = 24,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 dropout_path_rate: float = 0.0,
                 use_checkpoint: bool = False,
                 ):
        super().__init__(0, concat=concat, spatial_dims=spatial_dims)
        for _ in range(encoder_num):
            self.encoders.append(SwinTransformerEncoder(spatial_dims=spatial_dims,
                                                        input_channels=input_channels,
                                                        feature_size=feature_size,
                                                        depths=depths,
                                                        num_heads=num_heads,
                                                        drop_rate=drop_rate,
                                                        attn_drop_rate=attn_drop_rate,
                                                        dropout_path_rate=dropout_path_rate,
                                                        use_checkpoint=use_checkpoint,
                                                        ))
            self.out_channel = self.encoders[0].out_channel
            if lstm != 0 and spatial_dims == 2:
                self.rnns.append(TimeRNNAttentionPooling(
                    input_size=self.out_channel,
                    num_layers=lstm
                ))
        self.attn = MergeLinearAttention(in_features=self.out_channel, num_layers=encoder_num)
