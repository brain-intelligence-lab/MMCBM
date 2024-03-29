# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import torch.nn as nn
from models.backbone.blocks import BaseNet, CycleBaseNet, EfficientEncoder, SwinEncoder, DenseNetEncoder
from models.backbone.blocks import Classifier, PrognosisClassifier, RNNClassifier, AttnPoolClassifier, MaxPoolClassifier
from models.backbone.blocks import SingleBaseNet, MMDictClassifier, MMFusionClassifier
from models.backbone.blocks import VariationEfficientEncoder, Prognosis
from models.backbone.blocks import SingleEfficientEncoder, MMEfficientEncoder, FusionEfficientEncoder, \
    EfficientEncoderNew

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


class MMEfficientNet(BaseNet):
    def __init__(self, modality, num_layers, spatial_dims, input_channels=3, num_class=3, model_name='b2',
                 pretrained=True, rnn_type=nn.LSTM, bidirectional=False, rnn_attn=False, avg_pooling=True):
        if isinstance(input_channels, int):
            input_channels = {'FA': input_channels, 'ICGA': input_channels, 'US': input_channels}
        super(MMEfficientNet, self).__init__(
            modality=modality,
            Encoder_fn=lambda m, num_encoder: EfficientEncoder(model_name=efficientnet_params[model_name],
                                                               spatial_dims=spatial_dims[m],
                                                               input_channels=input_channels[m],
                                                               num_layers=num_layers[m],
                                                               concat=False,
                                                               pretrained=pretrained,
                                                               encoder_num=num_encoder,
                                                               bidirectional=bidirectional,
                                                               rnn_type=rnn_type,
                                                               rnn_attn=rnn_attn,
                                                               avg_pooling=avg_pooling),
            Classifier_fn=lambda out_channel, num_layers: Classifier(in_features=out_channel,
                                                                     out_features=num_class,
                                                                     attn=True,
                                                                     num_layers=num_layers),
            name='EfficientNet'
        )


class MMCycleEfficientNet(CycleBaseNet):
    def __init__(self, modality, num_layers, spatial_dims, input_channels=3, num_class=3, model_name='b2',
                 pretrained=True, rnn_type=nn.LSTM, bidirectional=False, rnn_attn=False):
        if isinstance(input_channels, int):
            input_channels = {'FA': input_channels, 'ICGA': input_channels, 'US': input_channels}
        super().__init__(
            modality=modality,
            Encoder_fn=lambda m, num_encoder: EfficientEncoder(model_name=efficientnet_params[model_name],
                                                               spatial_dims=spatial_dims[m],
                                                               input_channels=input_channels[m],
                                                               num_layers=num_layers[m],
                                                               concat=False,
                                                               pretrained=pretrained,
                                                               encoder_num=num_encoder,
                                                               bidirectional=bidirectional,
                                                               rnn_type=rnn_type,
                                                               rnn_attn=rnn_attn),
            name='EfficientNet',
            out_features=num_class
        )


class MMVariationEfficientNet(BaseNet):
    def __init__(self, modality, num_layers, input_channels=3, num_class=3, model_name='b2',
                 pretrained=True, rnn_type=nn.LSTM, istrain=True):
        if isinstance(input_channels, int):
            input_channels = {'FA': input_channels, 'ICGA': input_channels, 'US': input_channels}
        super().__init__(
            modality=modality,
            Encoder_fn=lambda m, num_encoder: VariationEfficientEncoder(model_name=efficientnet_params[model_name],
                                                                        input_channels=input_channels[m],
                                                                        num_layers=num_layers[m],
                                                                        pretrained=pretrained,
                                                                        encoder_num=num_encoder,
                                                                        rnn_type=rnn_type,
                                                                        istrain=istrain),
            Classifier_fn=lambda out_channel, num_layers: Classifier(in_features=out_channel,
                                                                     out_features=num_class,
                                                                     attn=True,
                                                                     num_layers=num_layers),
            name='EfficientNet',
            variation=True
        )


class MMDense121Net(BaseNet):
    def __init__(self, modality, num_layers, spatial_dims, input_channels=3, num_class=3, pretrained=True):
        if isinstance(input_channels, int):
            input_channels = {'FA': input_channels, 'ICGA': input_channels, 'US': input_channels}
        super().__init__(
            modality=modality,
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


class MMTransformer(BaseNet):
    def __init__(self, modality, lstm, concat, input_channels=3, num_class=3):
        super().__init__(
            modality=modality,
            Encoder_fn=lambda m, num_encoder: SwinEncoder(input_channels=input_channels,
                                                          depths=(2, 2, 2, 2),
                                                          num_heads=(3, 6, 12, 24),
                                                          lstm=lstm[m],
                                                          concat=concat,
                                                          encoder_num=num_encoder),
            Classifier_fn=lambda out_channel, num_layers: Classifier(in_features=out_channel,
                                                                     out_features=num_class,
                                                                     attn=True,
                                                                     num_layers=num_layers),
            name='Transformer',
        )


class MMPrognosis(Prognosis):
    def __init__(self, state_path, model, num_class=3, train_encoder=False):
        super().__init__(
            state_path=state_path,
            model=model,
            Classifier_fn=lambda out_channel, num_layers: PrognosisClassifier(in_features=out_channel,
                                                                              out_features=num_class,
                                                                              attn=True,
                                                                              num_layers=num_layers),
            name='Prognosis',
            train_encoder=train_encoder
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
                 pretrained=True, fusion='pool', avg_pooling=True):
        modalities = ['FA', 'ICGA', 'US']

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


class MMMaxSCLSEfficientNet(SingleBaseNet):
    def __init__(self, spatial_dims: int, input_channels: int = 3, num_class=3, model_name='b0',
                 pretrained=True, fusion='pool'):
        modalities = ['FA', 'ICGA', 'US']
        super(MMMaxSCLSEfficientNet, self).__init__(
            Encoder_fn=lambda **kwargs: MMEfficientEncoder(modalities=modalities,
                                                           model_name=efficientnet_params[model_name],
                                                           spatial_dims=spatial_dims,
                                                           input_channels=input_channels,
                                                           pretrained=pretrained,
                                                           ),
            Classifier_fn=lambda out_channel: MaxPoolClassifier(in_features=out_channel,
                                                                out_features=num_class)
            if fusion == 'pool' else RNNClassifier(in_features=out_channel,
                                                   out_features=num_class),
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


class MMFusionEfficient(SingleBaseNet):
    def __init__(self, num_layers, model_name='b2', spatial_dims: int = 2, input_channels: int = 3, num_class=3):
        super(MMFusionEfficient, self).__init__(
            Encoder_fn=lambda **kwargs: FusionEfficientEncoder(model_name=efficientnet_params[model_name],
                                                               spatial_dims=spatial_dims,
                                                               input_channels=input_channels),
            Classifier_fn=lambda out_channel: MMFusionClassifier(in_features=out_channel,
                                                                 out_features=num_class,
                                                                 num_layers=num_layers),
            name='Efficient'
        )


class MMFusionSingleCLSEfficient(SingleBaseNet):
    def __init__(self, num_layers, model_name='b2', spatial_dims: int = 2, input_channels: int = 3, num_class=3):
        super(MMFusionSingleCLSEfficient, self).__init__(
            Encoder_fn=lambda **kwargs: FusionEfficientEncoder(model_name=efficientnet_params[model_name],
                                                               spatial_dims=spatial_dims,
                                                               input_channels=input_channels,
                                                               single_cls=True),
            Classifier_fn=lambda out_channel: RNNClassifier(in_features=out_channel,
                                                            out_features=num_class,
                                                            num_layers=num_layers),
            name='Efficient'
        )


########################################## Efficient New MM Not Update Encoder ##########################################
class MMNUEEfficient(SingleBaseNet):
    def __init__(self, enc_no_grad, model_name='b2', num_layers=2, spatial_dims: int = 2, input_channels: int = 3,
                 num_class=3):
        super(MMNUEEfficient, self).__init__(
            Encoder_fn=lambda **kwargs: EfficientEncoderNew(model_name=efficientnet_params[model_name],
                                                            spatial_dims=spatial_dims,
                                                            input_channels=input_channels,
                                                            enc_no_grad=enc_no_grad),
            Classifier_fn=lambda out_channel: MMFusionClassifier(in_features=out_channel,
                                                                 out_features=num_class,
                                                                 num_layers=num_layers,
                                                                 mask_prob=0),
            name='Efficient'
        )


class MMNUESingleCLSEfficient(SingleBaseNet):
    def __init__(self, enc_no_grad, model_name='b2', num_layers=2, spatial_dims: int = 2, input_channels: int = 3,
                 num_class=3):
        super(MMNUESingleCLSEfficient, self).__init__(
            Encoder_fn=lambda **kwargs: EfficientEncoderNew(model_name=efficientnet_params[model_name],
                                                            spatial_dims=spatial_dims,
                                                            input_channels=input_channels,
                                                            enc_no_grad=enc_no_grad),
            Classifier_fn=lambda out_channel: RNNClassifier(in_features=out_channel,
                                                            out_features=num_class,
                                                            num_layers=num_layers),
            name='Efficient'
        )


if __name__ == '__main__':
    import torch

    x = torch.randn(1, 1, 128, 128, 128)
    y = torch.randn(1, 1, 3, 128, 128)
    num_layers = {'FA': 2, 'ICGA': 2, 'US': 0}
    model = MMEfficientNet('MM', spatial_dims={'FA': 3, 'ICGA': 3, 'US': 2},
                           input_channels={'FA': 1, 'ICGA': 1, 'US': 3},
                           num_layers=num_layers,
                           model_name='b2', pretrained=True, rnn_type=nn.LSTM, rnn_attn=True)
    o = model({'FA': x, 'ICGA': x, 'US': y}, 'MM')
    pass
