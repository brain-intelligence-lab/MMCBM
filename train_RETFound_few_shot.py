# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import torch.optim.lr_scheduler

from utils.logger import *
from monai.utils import set_determinism
from utils.metrics import *
from loss import Loss
from utils.decorator import decorator_args
from trainer.train_helper_backbone import TrainHelperBackbone


def get_model_opti(args):
    from few_shot.model import init_model
    model = init_model(args.device)
    args.transform = model.transform
    opt = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    return model, opt


@decorator_args
def get_args(args) -> argparse.Namespace:
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
    args.down_sample = False
    args.dir_name = f'RETFound'
    args.spatial_dims = 2
    args.clip_name = 'RET'
    args.lr = 1e-3
    args.bz = 8
    ###################### debug ######################
    args.extra_data = False
    ###################### debug ######################

    args.metrics = [
        Accuracy(),
        Precision(),
        Recall(),
        F1(),
    ]
    args.mode = 'max'


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    set_determinism(args.seed)
    model, opti = get_model_opti(args)
    args.loss = Loss(loss_type=args.loss)
    print(args)
    # start training
    TrainHelperBackbone(args, model, opti).start_train()
