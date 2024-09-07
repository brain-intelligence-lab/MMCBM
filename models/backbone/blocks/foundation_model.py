import torch
import torch.nn as nn

from .BaseNet import MMBaseEncoder
from few_shot.RETFound.models_vit import vit_large_patch16


class Backbone(nn.Module):
    def __init__(self, load_path='few_shot/RETFound/RETFound_cfp_weights.pth'):
        super(Backbone, self).__init__()
        self.encoder = vit_large_patch16(
            img_size=224,
            num_classes=5,
            drop_path_rate=0,
            global_pool=True,
        )
        checkpoint = torch.load(load_path, map_location='cpu')
        self.encoder.load_state_dict(checkpoint['model'], strict=False)
        self.encoder.eval()

    @property
    def out_channel(self):
        return self.encoder.embed_dim

    def forward(self, x):
        return self.encoder.forward_features(x)


class FoundationEncoder(MMBaseEncoder):
    def __init__(self, modalities, spatial_dims=2, pretrained: bool = True):
        super().__init__(out_channel=0, spatial_dims=spatial_dims)
        encoder = Backbone()
        self.encoder = nn.ModuleDict({modality: encoder for modality in modalities})
        for v in self.encoder.values():
            self.out_channel = v.out_channel
            break
