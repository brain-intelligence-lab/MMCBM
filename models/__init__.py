# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""
import os

from .CLIPs import MedClip, BioMedClip, Clip, OpenClip
from models.MMCBM.CBMs import MMLinearCBM
from models.backbone.MultiModels import MMAttnSCLSEfficientNet, MMFoundation
from utils.EarlyStop import EarlyStopping
from utils.dataloader import val_transforms
from params import img_size


def get_clip(clip_name, device, download_root, normalize):
    if 'med' in clip_name:
        clip_model = MedClip(device, clip_name)
    elif 'bio' in clip_name:
        clip_model = BioMedClip(device)
    elif 'open' in clip_name:
        clip_model = OpenClip(device, clip_name, normalize=normalize)
    else:
        clip_model = Clip(device, clip_name, 'cache_clip')
    return clip_model


def get_backbone(args):
    if 'clip' in args.clip_name or 'open' in args.clip_name:
        return get_clip(args.clip_name, args.device, 'clip_concepts_saved', not args.clip_nonorm)

    if 'foundation' in args.backbone:
        from few_shot.RETFound.utils import transform
        from utils.my_transformer import ClipTrans
        backbone = MMFoundation(
            model_name=args.model,
            fusion='pool',
            spatial_dims=2,
            num_class=args.out_channel,
            modalities=['FA', 'ICGA', 'US']
        )
        backbone.transform = ClipTrans(transform())
        backbone.transforms = ClipTrans(transform())
        args.transform = ClipTrans(transform())
    else:
        backbone = MMAttnSCLSEfficientNet(
            input_channels=3,
            model_name=args.model,
            fusion='pool',
            spatial_dims=2,
            num_class=args.out_channel,
            avg_pooling=True,
            modalities=['FA', 'ICGA', 'US']
        )
        if args.backbone is None:
            raise ValueError("Please specify the backbone")
        stoppers = EarlyStopping(dir=os.path.join('result/', f'{args.backbone}', f'epoch_stopper'),
                                 patience=args.patience,
                                 mode=args.mode)
        stoppers.load_checkpoint(backbone, ignore=True, strict=True,
                                 name=f'checkpoint_{args.bidx}.pth',
                                 cp=False)
        backbone.transform = val_transforms(True, img_size)
        backbone.transforms = val_transforms(True, img_size)
    backbone.to(args.device)
    backbone.eval()
    return backbone
