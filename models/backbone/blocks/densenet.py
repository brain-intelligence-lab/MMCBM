import torch.nn as nn

from monai.networks.layers import Pool, get_act_layer
from monai.networks.nets.densenet import _load_state_dict
from monai.networks.nets import DenseNet
from collections import OrderedDict
from .BaseNet import BaseEncoder


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
