# -*- encoding: utf-8 -*-
"""
@File    :   train_GIRNet_MIxInpMSE.py
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/6 9:12 PM   liu.yang      1.0         None
"""

import torch.optim.lr_scheduler

from utils.logger import *
from monai.utils import set_determinism
from utils.metrics import *
from utils.loss import *
from utils.decorator import decorator_args
from utils.base_train_single import start_train


def get_model_opti(args):
    from models.MultiModels import MMAttnSCLSEfficientNet
    model = MMAttnSCLSEfficientNet(
        input_channels=3,
        model_name=args.model,
        fusion=args.fusion,
        spatial_dims=args.spatial_dims,
        num_class=args.out_channel
    )
    opt = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    return model, opt


@decorator_args
def get_args(args):
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
    args.down_sample = False
    args.prognosis = False
    args.dir_name = f'Efficient{args.model}_SCLS'
    args.spatial_dims = 2
    ###################### debug ######################
    args.name = 'attnscls_CrossEntropy_32'
    args.epochs = 1
    args.plot = True
    args.infer = True
    args.resume = True
    args.device = 'cuda:0'
    args.bz = 1
    ###################### debug ######################

    args.metrics = [
        Accuracy(),
        Precision(),
        Precision('weighted'),
        Recall(),
        Recall('weighted'),
        F1(),
        F1('weighted')
    ]
    args.mode = 'max'


def main():
    if args.wandb:
        import wandb
        wandb.login()
    # args.model = 'b0'
    args.prognosis = False

    args.dir_name = f'Efficient{args.model}_SCLS'

    args.spatial_dims = 2
    args.kwargs = {}

    if 'Dummy' in args.name:
        args.out_channel = 4
    if 'SF' in args.name:
        args.time_shuffle = True
    if 'ave' in args.name:
        num = int(args.name.split('ave')[-1].split('_')[0])
        args.ave_sample = num

    if 'Focal' in args.name:
        args.loss = 'CrossFocal'
        from params import class_weight
        args.kwargs = {'gamma': 2, 'class_weight': class_weight}

    if 'Dice' in args.name:
        args.loss = 'DiceLoss'
        from params import class_weight
        args.kwargs = {'class_weight': class_weight}

    if 'GHM' in args.name:
        args.loss = 'MultiGHMC_Loss'
        args.kwargs = {'bins': 10, 'alpha': 0.5}

    if 'Smooth' in args.name:
        args.kwargs.update({'label_smoothing': .1})

    args.loss = Loss(loss_type=args.loss, **args.kwargs)
    print(args)
    args.metrics = [
        F1(),
        F1('weighted'),
        Accuracy(),
        Precision(),
        Precision('weighted'),
        Recall(),
        Recall('weighted'),
    ]
    args.mode = 'max'
    set_determinism(args.seed)
    model, opti = get_model_opti(args)
    start_train(args, model, opti)


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    set_determinism(args.seed)
    model, opti = get_model_opti(args)
    args.loss = Loss(loss_type=args.loss)
    print(args)
    # start training
    start_train(args, model, opti)
