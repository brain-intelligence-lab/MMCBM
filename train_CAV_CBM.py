# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""
import os
import argparse

import torch.optim.lr_scheduler

from utils.EarlyStop import EarlyStopping
from monai.utils import set_determinism
from utils.metrics import *
from loss import Loss
from utils.decorator import decorator_args
from params import id_to_labels
from trainer.train_helper_mmcbm import TrainHelperMMCBM
from models.MMCBM.concepts_bank import ConceptBank
from models import get_backbone


def get_model_opti(args):
    backbone = get_backbone(args)
    bank_dir = os.path.join(args.output_dir, args.dir_name)
    # initialize the concept bank
    args.concept_bank = ConceptBank(device=args.device,
                                    clip_name=args.clip_name,
                                    location=args.cbm_location,
                                    backbone=backbone,
                                    n_samples=args.pos_samples,
                                    neg_samples=args.neg_samples,
                                    svm_C=args.svm_C,
                                    bank_dir=bank_dir,
                                    report_shot=args.report_shot,
                                    concept_shot=args.concept_shot,
                                    cav_split=args.cav_split,
                                    language='zh'
                                    )
    from models.MMCBM.CBMs import M2LinearCBM
    # initialize the Concept Bottleneck Model: FA_ICGA and US
    model = M2LinearCBM(
        idx_to_class=id_to_labels,
        concept_bank=args.concept_bank,
        n_classes=args.out_channel,
        fusion=args.fusion,
        activation=args.activation,
        analysis_top_k=args.analysis_top_k,
        analysis_threshold=args.analysis_threshold,
        act_on_weight=args.act_on_weight,
        init_method=args.init_method,
        bias=args.bias,
    )
    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    return model, opt


@decorator_args
def get_args(args) -> argparse.Namespace:
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
    args.down_sample = False
    ##################### debug #####################
    # args.device = 0
    # args.k = 0
    args.wandb = False
    # args.clip_name = 'clip_ViT-L/14'
    args.cbm_model = 'b0'
    args.modality = 'MM'
    args.fusion = 'max'
    args.cbm_location = 'report_strict'
    args.analysis_top_k = 15
    args.activation = 'sigmoid'
    args.act_on_weight = True
    args.num_worker = 2
    # args.backbone = 'Efficientb0_SCLS_TestOnly/fold_0'
    ##################### debug #####################
    args.metrics = [
        Accuracy(),
        Precision(),
        Recall(),
        F1(),
    ]
    args.mode = 'max'


if __name__ == "__main__":
    from utils.logger import PrintColor
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    set_determinism(args.seed)
    model, opti = get_model_opti(args)
    args.loss = Loss(loss_type=args.loss, model=model)
    # start training
    TrainHelperMMCBM(args, model, opti).start_train()
