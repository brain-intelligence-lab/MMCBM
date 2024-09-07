# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

from models.backbone.blocks import BaseNet, SingleBaseNet
from models.backbone.blocks import Classifier, RNNClassifier, AttnPoolClassifier, MaxPoolClassifier, MMDictClassifier
from models.backbone.blocks import DenseNetEncoder, SingleEfficientEncoder, MMEfficientEncoder

efficientnet_params = {
    "b0": "efficientnet-b0",
    "b1": "efficientnet-b1",
    "b2": "efficientnet-b2",
    "b3": "efficientnet-b3",
    "b4": "efficientnet-b4",
    "b5": "efficientnet-b5",
    "b6": "efficientnet-b6",
    "b7": "efficientnet-b7",
    "b8": "efficientnet-b8",
    "l2": "efficientnet-l2",
}


class MMDense121Net(SingleBaseNet):
    def __init__(self, num_layers, spatial_dims, input_channels=3, num_class=3, pretrained=True):
        if isinstance(input_channels, int):
            input_channels = {'FA': input_channels, 'ICGA': input_channels, 'US': input_channels}
        super().__init__(
            Encoder_fn=lambda m, num_encoder: DenseNetEncoder(
                spatial_dims=spatial_dims[m],
                input_channels=input_channels[m],
                num_layers=num_layers[m],
                pretrained=pretrained,
                encoder_num=num_encoder,
            ),
            Classifier_fn=lambda out_channel, num_layers: Classifier(in_features=out_channel,
                                                                     out_features=num_class,
                                                                     attn=True,
                                                                     num_layers=num_layers),
            name='DenseNet'
        )


########################################## Single FusionEncoder MM ##########################################

class MMSingleEfficientNet(SingleBaseNet):
    def __init__(self, num_layers, spatial_dims: int, input_channels: int = 3, num_class=3, model_name='b3',
                 pretrained=True):
        super(MMSingleEfficientNet, self).__init__(
            Encoder_fn=lambda **kwargs: SingleEfficientEncoder(model_name=efficientnet_params[model_name],
                                                               spatial_dims=spatial_dims,
                                                               input_channels=input_channels,
                                                               pretrained=pretrained,
                                                               ),
            Classifier_fn=lambda out_channel: RNNClassifier(in_features=out_channel,
                                                            out_features=num_class,
                                                            num_layers=num_layers),
            name='EfficientNet'
        )


########################################## Single CLS ##########################################


class MMSCLSEfficientNet(SingleBaseNet):
    def __init__(self, num_layers, spatial_dims: int, input_channels: int = 3,
                 num_class=3, model_name='b3', pretrained=True):
        modalities = ['FA', 'ICGA', 'US']
        super(MMSCLSEfficientNet, self).__init__(
            Encoder_fn=lambda **kwargs: MMEfficientEncoder(modalities=modalities,
                                                           model_name=efficientnet_params[model_name],
                                                           spatial_dims=spatial_dims,
                                                           input_channels=input_channels,
                                                           pretrained=pretrained,
                                                           ),
            Classifier_fn=lambda out_channel: RNNClassifier(in_features=out_channel,
                                                            out_features=num_class,
                                                            num_layers=num_layers),
            name='EfficientNet'
        )


class MMAttnSCLSEfficientNet(SingleBaseNet):
    def __init__(self, spatial_dims: int, input_channels: int = 3, num_class=3, model_name='b0',
                 pretrained=True, fusion='pool', avg_pooling=True, modalities=['FA', 'ICGA', 'US']):
        self.mm_order = modalities

        def get_cls(out_channel):
            if fusion == 'pool':
                return AttnPoolClassifier(in_features=out_channel,
                                          out_features=num_class)
            elif fusion == 'lstm':
                return RNNClassifier(in_features=out_channel,
                                     out_features=num_class)
            elif fusion == 'max':
                return MaxPoolClassifier(in_features=out_channel,
                                         out_features=num_class)
            else:
                raise ValueError(f'No such fusion: {fusion}')

        super(MMAttnSCLSEfficientNet, self).__init__(
            Encoder_fn=lambda **kwargs: MMEfficientEncoder(modalities=modalities,
                                                           model_name=efficientnet_params[model_name],
                                                           spatial_dims=spatial_dims,
                                                           input_channels=input_channels,
                                                           pretrained=pretrained,
                                                           avg_pooling=avg_pooling
                                                           ),
            Classifier_fn=get_cls,
            name=f'EfficientNet{model_name}'
        )


class MMFusionResNet(SingleBaseNet):
    def __init__(self, num_layers, spatial_dims: int, input_channels: int = 3, num_class=3):
        from models.backbone.blocks.efficientnet import ResNetEncoder
        super(MMFusionResNet, self).__init__(
            Encoder_fn=lambda **kwargs: ResNetEncoder(spatial_dims=spatial_dims,
                                                      input_channels=input_channels),
            Classifier_fn=lambda out_channel: MMDictClassifier(in_features=out_channel,
                                                               out_features=num_class,
                                                               num_layers=num_layers),
            name='ResNet'
        )


########################################## Using Foundation model ##########################################
from .blocks.foundation_model import FoundationEncoder


class MMFoundation(SingleBaseNet):
    def __init__(self, spatial_dims: int, num_class=3, model_name='foundation',
                 pretrained=True, fusion='pool', modalities=['FA', 'ICGA', 'US']):
        self.mm_order = modalities

        def get_cls(out_channel):
            if fusion == 'pool':
                return AttnPoolClassifier(in_features=out_channel,
                                          out_features=num_class)
            elif fusion == 'max':
                return MaxPoolClassifier(in_features=out_channel,
                                         out_features=num_class)
            else:
                raise ValueError(f'No such fusion: {fusion}')

        super(MMFoundation, self).__init__(
            Encoder_fn=lambda **kwargs: FoundationEncoder(modalities=modalities,
                                                          spatial_dims=spatial_dims,
                                                          pretrained=pretrained,
                                                          ),
            Classifier_fn=get_cls,
            name=f'Foundation{model_name}'
        )
