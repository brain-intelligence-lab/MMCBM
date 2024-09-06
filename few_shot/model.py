import torch
import torch.nn as nn
from .RETFound.models_vit import vit_large_patch16
from .RETFound.utils import transform
from utils.my_transformer import ClipTrans


def classification_block(inp_dim, out_dim, dropout=0.0, relu=False) -> nn.Sequential:
    """Creates a classification block.
    https://github.com/sergiuoprea/clip_with_few_shots/tree/master/project

    Args:
        inp_dim ([type]): input dimension to the block.
        out_dim ([type]): output dimension to the block.
        dropout (float, optional): if greater than 0.0, adds a dropout layer with
            given probability. Defaults to 0.0.
        batch_norm (bool, optional): if true, adds a batchnorm layer. Defaults to False.

    Returns:
        [nn.Sequential]: block combining Linear + BatchNorm + ReLU + Dropout
    """
    layers = []

    layers.append(nn.Linear(in_features=inp_dim, out_features=out_dim))

    if relu:
        layers.append(nn.ReLU(inplace=True))

    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)


class BasicClassifier(nn.Module):
    def __init__(self, input_dim, out_classes=3):
        super(BasicClassifier, self).__init__()
        # self.block0 = classification_block(input_dim, input_dim // 2, dropout=0.2, relu=True)
        self.block1 = classification_block(input_dim, out_classes)

    def forward(self, x):
        # x = self.block0(x)
        x = self.block1(x)
        return x


class FewShotModel(nn.Module):
    def __init__(self, backbone, n_classes, transform, multi=True):
        super(FewShotModel, self).__init__()
        self.backbone = lambda x: backbone.forward_features(x)
        self.n_classes = n_classes
        self.transform = transform
        self.multi = multi
        self.classifier = BasicClassifier(input_dim=backbone.embed_dim, out_classes=n_classes)

    def to_B(self, x):
        if x.ndim == 5:
            self.time_step = x.shape[1]
            x = x.reshape(-1, *x.shape[2:])
        return x

    def to_T(self, x):
        if self.time_step is not None:
            x = x.reshape(-1, self.time_step, x.shape[-1])
            self.time_step = None
        return x

    def forward_mm(self, x, modality):
        if 'MM' not in modality:
            x = {modality: x[modality]}
        embd = []
        with torch.no_grad():
            for modality, img in x.items():
                img = self.to_B(img)
                img = self.backbone(img)
                img = self.to_T(img)
                embd.append(img)
        embd = torch.cat(embd, dim=1).max(dim=1).values
        embd = self.classifier(embd)
        return embd

    def forward_single(self, x, modality):
        if 'MM' in modality:
            x = {modality: x[modality]}
        embd = self.backbone(x[modality])
        embd = self.classifier(embd)
        return embd

    def forward(self, x, modality):
        if self.multi:
            return self.forward_mm(x, modality)
        else:
            return self.forward_single(x, modality)


def init_model(device):
    backbone = vit_large_patch16(
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
    )
    # load model
    checkpoint = torch.load('/data_smr/liuy/Project/MMCBM/few_shot/RETFound/RETFound_cfp_weights.pth',
                            map_location='cpu')
    backbone.load_state_dict(checkpoint['model'], strict=False)
    backbone.eval()
    backbone.to(device)
    model = FewShotModel(backbone, n_classes=3, transform=ClipTrans(transform()))
    return model
