import torch
import numpy as np
from . import models_vit
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def transform():
    return Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=imagenet_mean, std=imagenet_std)
    ])


def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    # build model
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
    )
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    # build transforms

    return model, transform()


def run_one_image(img, model, device):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    x = x.to(device, non_blocking=True)
    latent = model.forward_features(x.float())
    latent = torch.squeeze(latent)

    return latent
