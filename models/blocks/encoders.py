import torch
import torch.nn as nn

from monai.networks.nets.resnet import get_avgpool
from functools import partial
from monai.networks.layers import Norm, Conv, get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.networks.layers import Pool

from .layers import MAModule, TimeRNNAttentionPooling, MergeLinearAttention
from .BaseNet import MMBaseEncoder, SingleBaseEncoder, BaseEncoder, VariationEncoder


class FusionEncoder(nn.Module):
    def __init__(self, modality=None, encoders=None, *args, **kwargs):
        super().__init__()
        self.modality = modality
        self.encoders: [FusionEncoder] = encoders
        if encoders is not None:
            self.ma = MAModule()

    def head(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def output(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_layers(self) -> list:
        pass

    def get_other_encoder_layers_output(self, x: dict):
        layers_output = []
        if self.encoders is not None:
            max_b = max([v.shape[0] for v in x.values()])
            min_b = min([v.shape[0] for v in x.values()])
            for key in x.keys():
                if x[key].shape[0] != max_b:
                    x[key] = x[key].unsqueeze(1).repeat(1, max_b // min_b, 1, 1, 1).view(max_b, *x[key].shape[1:])
            with torch.no_grad():
                max_b = 0
                for encoder in self.encoders:
                    out = []
                    encoder(x, hiddens=out)
                    if out[0].shape[0] > max_b:
                        max_b = out[0].shape[0]
                    out = [o.cpu() for o in out]
                    layers_output.append(out)
        return list(zip(*layers_output))

    def fusion(self, x, fusion_outputs):
        fusion_outputs = [fo.to(x.device) for fo in fusion_outputs]
        return self.ma(torch.stack([x, *fusion_outputs], dim=1)) + x

    def forward(self, x: dict, hiddens=None):
        fusion_outputs = self.get_other_encoder_layers_output(x)
        if len(fusion_outputs) == 0:
            x = x[self.modality] if isinstance(x, dict) else x
            x = self.head(x)
        else:
            device = list(x.values())[0].device
            x = torch.zeros_like(fusion_outputs[0][0]).to(device)

        for idx, layer in enumerate(self.get_layers()):
            if isinstance(hiddens, list):
                hiddens.append(x.clone())

            if len(fusion_outputs) > 0:
                x = self.fusion(x, fusion_outputs[idx])
            x = layer(x)
        x = self.output(x)
        return x


class MMFusionEncoder(MMBaseEncoder):
    def __init__(self, out_channel, spatial_dims, single_cls=False):
        super().__init__(out_channel=out_channel, spatial_dims=spatial_dims)
        self.single_cls = single_cls

    def __getitem__(self, item):
        if item == 'MM':
            return self.mm_encoder
        return self.encoder[item]

    def modality_forward(self, x):
        out = {}
        for m in x.keys():
            v = self.encoder[m](x)
            out[m] = v
        out = {m: self.to_T(v) for m, v in out.items()}
        return out

    def forward(self, x: dict):
        x = {m: self.to_B(v) for m, v in x.items()}
        if len(x) == 3:
            if self.single_cls:
                with torch.no_grad():
                    out = self.modality_forward(x)
                out.update({'MM': self.to_T(self.mm_encoder(x))})
                return out
            return {'MM': self.to_T(self.mm_encoder(x))}
        return self.modality_forward(x)


######################################### ResNet #########################################

class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes: int,
            planes: int,
            spatial_dims: int = 3,
            stride: int = 1,
            downsample=None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            in_planes: int,
            planes: int,
            spatial_dims: int = 3,
            stride: int = 1,
            downsample=None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: which downsample layer to use.
        """

        super().__init__()

        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(FusionEncoder):
    """
    ResNet based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? <https://arxiv.org/pdf/1711.09577.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.

    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
            ResNet block class or str.
            for Basic: ResNetBlock or 'basic'
            for Bottleneck: ResNetBottleneck or 'bottleneck'
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.

    """

    def __init__(
            self,
            block,
            layers,
            block_inplanes,
            spatial_dims: int = 3,
            n_input_channels: int = 3,
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool: bool = False,
            shortcut_type: str = "B",
            widen_factor: float = 1.0,
            modality=None,
            encoders=None
    ) -> None:

        super().__init__(modality, encoders)

        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type = Norm[Norm.BATCH, spatial_dims]
        pool_type = Pool[Pool.MAX, spatial_dims]
        avgp_type = Pool[Pool.ADAPTIVEAVG, spatial_dims]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,  # type: ignore
            stride=conv1_stride,  # type: ignore
            padding=tuple(k // 2 for k in conv1_kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
            self,
            block,
            planes: int,
            blocks: int,
            spatial_dims: int,
            shortcut_type: str,
            stride: int = 1,
    ) -> nn.Sequential:

        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_type = Norm[Norm.BATCH, spatial_dims]

        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                    norm_type(planes * block.expansion),
                )

        layers = [
            block(in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims,
                  stride=stride, downsample=downsample
                  )
        ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims))

        return nn.Sequential(*layers)

    def head(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

    def output(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_layers(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]


class ResNetEncoder(MMFusionEncoder):
    def __init__(self,
                 modalities=None,
                 spatial_dims=2,
                 input_channels=3,
                 block=ResNetBlock,
                 layers=(2, 2, 2, 2),
                 block_inplanes=(32, 64, 128, 256)
                 ):
        super(ResNetEncoder, self).__init__(out_channel=block_inplanes[3] * block.expansion, spatial_dims=spatial_dims)
        if modalities is None:
            modalities = ['US', 'ICGA', 'FA']
        encoders = [ResNet(modality=m, spatial_dims=2, block=block, n_input_channels=input_channels, layers=layers,
                           block_inplanes=block_inplanes)
                    for m in modalities]
        self.encoder = nn.ModuleDict({en.modality: en for en in encoders})
        self.mm_encoder = ResNet(modality='MM',
                                 spatial_dims=2,
                                 block=block,
                                 n_input_channels=input_channels,
                                 layers=layers,
                                 block_inplanes=block_inplanes,
                                 encoders=encoders)


##################################################### Efficient #######################################################

from monai.networks.nets.efficientnet import _round_filters, BlockArgs, _make_same_padder, \
    get_norm_layer, _calculate_output_image_size, _round_repeats, MBConvBlock, Act, reduce, operator, math, \
    look_up_option, _load_state_dict

efficientnet_params = {
    # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.2),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.2),
}
blocks_args_str = [
    "r1_k3_s11_e1_i32_o16_se0.25",
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]


class EfficientNet(FusionEncoder):

    def __init__(
            self,
            model_name,
            spatial_dims: int = 2,
            in_channels: int = 3,
            norm=("batch", {"eps": 1e-3, "momentum": 0.01}),
            depth_divisor: int = 8,
            modality=None,
            encoders=None,
            pretrained=True
    ) -> None:
        """
        EfficientNet based on `Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/pdf/1905.11946.pdf>`_.
        Adapted from `EfficientNet-PyTorch <https://github.com/lukemelas/EfficientNet-PyTorch>`_.

        Args:
            blocks_args_str: block definitions.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            width_coefficient: width multiplier coefficient (w in paper).
            depth_coefficient: depth multiplier coefficient (d in paper).
            dropout_rate: dropout rate for dropout layers.
            image_size: input image resolution.
            norm: feature normalization type and arguments.
            drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.
            depth_divisor: depth divisor for channel rounding.

        """
        super().__init__(modality=modality, encoders=encoders)

        # check if model_name is valid model
        if model_name not in efficientnet_params:
            model_name_string = ", ".join(efficientnet_params.keys())
            raise ValueError(f"invalid model_name {model_name} found, must be one of {model_name_string} ")

        # get network parameters
        width_coefficient, depth_coefficient, image_size, dropout_rate, drop_connect_rate = efficientnet_params[
            model_name]

        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims can only be 1, 2 or 3.")

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        conv_type = Conv["conv", spatial_dims]
        adaptivepool_type = Pool[
            "adaptiveavg", spatial_dims
        ]

        # decode blocks args into arguments for MBConvBlock
        blocks_args = [BlockArgs.from_string(s) for s in blocks_args_str]

        # checks for successful decoding of blocks_args_str
        if not isinstance(blocks_args, list):
            raise ValueError("blocks_args must be a list")

        if blocks_args == []:
            raise ValueError("block_args must be non-empty")

        self._blocks_args = blocks_args
        self.in_channels = in_channels
        self.drop_connect_rate = drop_connect_rate

        # expand input image dimensions to list
        current_image_size = [image_size] * spatial_dims

        # Stem
        stride = 2
        out_channels = _round_filters(32, width_coefficient, depth_divisor)  # number of output channels
        self._conv_stem = conv_type(self.in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self._conv_stem_padding = _make_same_padder(self._conv_stem, current_image_size)
        self._bn0 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)
        current_image_size = _calculate_output_image_size(current_image_size, stride)

        # build MBConv blocks
        num_blocks = 0
        self._blocks = nn.Sequential()

        self.extract_stacks = []

        # update baseline blocks to input/output filters and number of repeats based on width and depth multipliers.
        for idx, block_args in enumerate(self._blocks_args):
            block_args = block_args._replace(
                input_filters=_round_filters(block_args.input_filters, width_coefficient, depth_divisor),
                output_filters=_round_filters(block_args.output_filters, width_coefficient, depth_divisor),
                num_repeat=_round_repeats(block_args.num_repeat, depth_coefficient),
            )
            self._blocks_args[idx] = block_args

            # calculate the total number of blocks - needed for drop_connect estimation
            num_blocks += block_args.num_repeat

            if block_args.stride > 1:
                self.extract_stacks.append(idx)

        self.extract_stacks.append(len(self._blocks_args))

        # create and add MBConvBlocks to self._blocks
        idx = 0  # block index counter
        for stack_idx, block_args in enumerate(self._blocks_args):
            blk_drop_connect_rate = self.drop_connect_rate

            # scale drop connect_rate
            if blk_drop_connect_rate:
                blk_drop_connect_rate *= float(idx) / num_blocks

            sub_stack = nn.Sequential()
            # the first block needs to take care of stride and filter size increase.
            sub_stack.add_module(
                str(idx),
                MBConvBlock(
                    spatial_dims=spatial_dims,
                    in_channels=block_args.input_filters,
                    out_channels=block_args.output_filters,
                    kernel_size=block_args.kernel_size,
                    stride=block_args.stride,
                    image_size=current_image_size,
                    expand_ratio=block_args.expand_ratio,
                    se_ratio=block_args.se_ratio,
                    id_skip=block_args.id_skip,
                    norm=norm,
                    drop_connect_rate=blk_drop_connect_rate,
                ),
            )
            idx += 1  # increment blocks index counter

            current_image_size = _calculate_output_image_size(current_image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            # add remaining block repeated num_repeat times
            for _ in range(block_args.num_repeat - 1):
                blk_drop_connect_rate = self.drop_connect_rate

                # scale drop connect_rate
                if blk_drop_connect_rate:
                    blk_drop_connect_rate *= float(idx) / num_blocks

                # add blocks
                sub_stack.add_module(
                    str(idx),
                    MBConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_args.input_filters,
                        out_channels=block_args.output_filters,
                        kernel_size=block_args.kernel_size,
                        stride=block_args.stride,
                        image_size=current_image_size,
                        expand_ratio=block_args.expand_ratio,
                        se_ratio=block_args.se_ratio,
                        id_skip=block_args.id_skip,
                        norm=norm,
                        drop_connect_rate=blk_drop_connect_rate,
                    ),
                )
                idx += 1  # increment blocks index counter

            self._blocks.add_module(str(stack_idx), sub_stack)

        # sanity check to see if len(self._blocks) equal expected num_blocks
        if idx != num_blocks:
            raise ValueError("total number of blocks created != num_blocks")

        # Head
        head_in_channels = block_args.output_filters
        out_channels = _round_filters(1280, width_coefficient, depth_divisor)
        self._conv_head = conv_type(head_in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_head_padding = _make_same_padder(self._conv_head, current_image_size)
        self._bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)

        # final linear layer
        self._avg_pooling = adaptivepool_type(1)
        self._dropout = nn.Dropout(dropout_rate)
        # self._fc = nn.Linear(out_channels, self.num_classes)
        self.out_channels = out_channels
        # swish activation to use - using memory efficient swish by default
        # can be switched to normal swish using self.set_swish() function call
        self._swish = Act["memswish"]()

        # initialize weights using Tensorflow's init method from official impl.
        self._initialize_weights()

        # only pretrained for when `spatial_dims` is 2
        if pretrained and (spatial_dims == 2):
            _load_state_dict(self, model_name, True, False)

    def set_swish(self, memory_efficient: bool = True) -> None:
        """
        Sets swish function as memory efficient (for training) or standard (for JIT export).

        Args:
            memory_efficient: whether to use memory-efficient version of swish.

        """
        self._swish = Act["memswish"]() if memory_efficient else Act["swish"](alpha=1.0)
        for sub_stack in self._blocks:
            for block in sub_stack:
                block.set_swish(memory_efficient)

    def head(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        return x

    def output(self, x: torch.Tensor) -> torch.Tensor:
        # Head
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))

        # Pooling and final linear layer
        x = self._avg_pooling(x)

        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        return x

    def get_layers(self) -> list:
        return list(self._blocks)

    def _initialize_weights(self) -> None:
        """
        Args:
            None, initializes weights for conv/linear/batchnorm layers
            following weight init methods from
            `official Tensorflow EfficientNet implementation
            <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L61>`_.
            Adapted from `EfficientNet-PyTorch's init method
            <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py>`_.
        """
        for _, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                fan_out = reduce(operator.mul, m.kernel_size, 1) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                fan_in = 0
                init_range = 1.0 / math.sqrt(fan_in + fan_out)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()


class EfficientEncoder(BaseEncoder):
    def __init__(self, model_name, spatial_dims=2, input_channels=3, num_layers=0, concat=False,
                 pretrained: bool = True, encoder_num=1,
                 bidirectional=False, rnn_type=nn.LSTM, rnn_attn=False):
        super(EfficientEncoder, self).__init__(out_channel=0, concat=concat, spatial_dims=spatial_dims)
        for _ in range(encoder_num):
            self.encoders.append(EfficientNet(model_name=model_name, spatial_dims=spatial_dims,
                                              in_channels=input_channels, pretrained=pretrained))
            self.out_channel = self.encoders[0].out_channels
            if num_layers != 0 and spatial_dims == 2:
                self.rnns.append(
                    rnn_type(input_size=self.out_channel,
                             hidden_size=self.out_channel,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=bidirectional) if not rnn_attn else TimeRNNAttentionPooling(
                        input_size=self.out_channel,
                        num_layers=num_layers)
                )
        in_features = self.out_channel * 2 if bidirectional and num_layers != 0 and spatial_dims == 2 else self.out_channel
        self.attn = MergeLinearAttention(in_features=in_features,
                                         out_features=self.out_channel,
                                         num_layers=encoder_num)


class VariationEfficientEncoder(VariationEncoder):
    def __init__(self, model_name, input_channels=3, num_layers=0, pretrained: bool = True, encoder_num=1,
                 rnn_type=nn.LSTM, istrain=True):
        super().__init__(istrain=istrain)
        for _ in range(encoder_num):
            self.encoders.append(EfficientNet(model_name=model_name, in_channels=input_channels, pretrained=pretrained))
            self.out_channel = 512
            if num_layers != 0:
                self.rnns.append(
                    rnn_type(
                        input_size=self.encoders[0].out_channels,
                        hidden_size=512,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=False
                    )
                )
            in_features = self.out_channel if num_layers != 0 else self.encoders[0].out_channels
            self.variations.append(
                nn.Linear(in_features=in_features, out_features=self.out_channel * 2)
            )


class FusionEfficientEncoder(MMFusionEncoder):
    def __init__(self,
                 modalities=None,
                 spatial_dims=2,
                 input_channels=3,
                 model_name='efficientnet-b0',
                 pretrained=True,
                 single_cls=False
                 ):
        super(FusionEfficientEncoder, self).__init__(out_channel=1280,
                                                     spatial_dims=spatial_dims,
                                                     single_cls=single_cls)
        if modalities is None:
            modalities = ['US', 'ICGA', 'FA']
        encoders = [EfficientNet(modality=m,
                                 spatial_dims=2,
                                 model_name=model_name,
                                 in_channels=input_channels,
                                 pretrained=pretrained) for m in modalities
                    ]
        self.encoder = nn.ModuleDict({en.modality: en for en in encoders})
        self.mm_encoder = EfficientNet(modality='MM',
                                       spatial_dims=2,
                                       model_name=model_name,
                                       in_channels=input_channels,
                                       pretrained=pretrained,
                                       encoders=encoders)


class EfficientEncoderNew(MMBaseEncoder):
    def __init__(self, model_name, spatial_dims=2, input_channels=3, pretrained: bool = True, enc_no_grad=False):
        super(EfficientEncoderNew, self).__init__(out_channel=1280, spatial_dims=spatial_dims, enc_no_grad=enc_no_grad)
        self.encoder = nn.ModuleDict({
            m: EfficientNet(modality=m,
                            spatial_dims=2,
                            model_name=model_name,
                            in_channels=input_channels,
                            pretrained=pretrained) for m in ['US', 'ICGA', 'FA']
        })

    def __getitem__(self, item):
        if 'MM' == item:
            return self.encoder
        return self.encoder[item]


########################################## Single Encoders ##########################################


class SingleEfficientEncoder(SingleBaseEncoder):
    def __init__(self, model_name, spatial_dims=2, input_channels=3,
                 pretrained: bool = True):
        super(SingleEfficientEncoder, self).__init__(out_channel=0, spatial_dims=spatial_dims)
        self.encoder = EfficientNet(model_name=model_name, spatial_dims=spatial_dims,
                                    in_channels=input_channels, pretrained=pretrained)
        self.out_channel = self.encoder.out_channels


############################################## Multi FusionEncoder ##############################################


class MMEfficientEncoder(MMBaseEncoder):
    def __init__(self, modalities, model_name, spatial_dims=2, input_channels=3,
                 pretrained: bool = True):
        super(MMEfficientEncoder, self).__init__(out_channel=0, spatial_dims=spatial_dims)
        self.encoder = nn.ModuleDict({
            m: EfficientNet(model_name=model_name, spatial_dims=spatial_dims,
                            in_channels=input_channels, pretrained=pretrained)
            for m in modalities
        })
        for v in self.encoder.values():
            self.out_channel = v.out_channels
            break
