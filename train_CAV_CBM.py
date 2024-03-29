# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import torch.optim.lr_scheduler

from utils.EarlyStop import EarlyStopping
from utils.logger import *
from monai.utils import set_determinism
from utils.metrics import *
from loss import Loss
from utils.decorator import decorator_args
from params import id_to_labels
from trainer.train_helper_mmcbm import TrainHelperMMCBM
from models.MMCBM.concepts_bank import ConceptBank


def get_model_opti(args):
    if 'clip' in args.clip_name:
        backbone = None
    else:
        from models.backbone.MultiModels import MMAttnSCLSEfficientNet
        backbone = MMAttnSCLSEfficientNet(
            input_channels=3,
            model_name=args.model,
            fusion='pool',
            spatial_dims=2,
            num_class=args.out_channel,
            avg_pooling=True,
        )
        if args.backbone is None:
            raise ValueError("Please specify the backbone")
        stoppers = EarlyStopping(dir=os.path.join('result/', f'{args.backbone}', f'epoch_stopper'),
                                 patience=args.patience,
                                 mode=args.mode)
        stoppers.load_checkpoint(backbone, ignore=True, strict=True,
                                 name=f'checkpoint_{args.idx}.pth' if int(args.idx) > 0 else 'MM.pth',
                                 cp=False)
        backbone.to(args.device)
        backbone.eval()
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
                                    )
    from models.MMCBM.CBMs import MMLinearCBM, SLinearCBM, M2LinearCBM
    if args.cbm_model == 'mm':
        # initialize the Concept Bottleneck Model
        model = MMLinearCBM(
            idx_to_class=id_to_labels,
            concept_bank=args.concept_bank,
            n_classes=args.out_channel,
            fusion=args.fusion,
            activation=args.activation,
            analysis_top_k=args.analysis_top_k,
            analysis_threshold=args.analysis_threshold,
            act_on_weight=args.act_on_weight,
            init_method=args.init_method,
            bias=args.bias)
    elif args.cbm_model == 'm2':
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
    elif args.cbm_model == 's':
        model = SLinearCBM(
            idx_to_class=id_to_labels,
            concept_bank=args.concept_bank,
            n_classes=args.out_channel,
            fusion=args.fusion,
            activation=args.activation,
            analysis_top_k=args.analysis_top_k,
            analysis_threshold=args.analysis_threshold,
            act_on_weight=args.act_on_weight,
            init_method=args.init_method,
            modality_mask=args.modality_mask,
            bias=args.bias,
        )
    elif args.cbm_model == 'sa':
        from models.MMCBM.CBMs import SALinearCBM
        model = SALinearCBM(
            idx_to_class=id_to_labels,
            concept_bank=args.concept_bank,
            n_classes=args.out_channel,
            fusion=args.fusion,
            activation=args.activation,
            analysis_top_k=args.analysis_top_k,
            analysis_threshold=args.analysis_threshold,
            modality_mask=args.modality_mask,
            bias=args.bias,
        )
    else:
        raise ValueError(f'Unknown cbm model {args.cbm_model}')
    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    return model, opt


@decorator_args
def get_args(args) -> argparse.Namespace:
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
    args.down_sample = False
    args.prognosis = False
    ##################### debug #####################
    # args.device = 0
    # args.wandb = True
    # args.clip_name = 'cav'
    # args.cbm_model = 'occ'
    # args.modality = 'MM'
    # args.name = 'OccCBM'
    # args.fusion = 'max'
    # args.mark = f'{args.cbm_location}_{args.mark}'
    # args.cbm_location = 'report_strict'
    # args.infer = False
    # args.resume = False
    # args.activation = 'sigmoid'
    # args.act_on_weight = True
    # args.idx = 180
    # args.num_worker = 0
    # args.backbone = 'Efficientb0_SCLS_attnscls_CrossEntropy_32/fold_0'
    ##################### debug #####################
    if 'clip' in args.clip_name:
        args.dir_name = f'CLip'
    else:
        args.dir_name = f'CAV'

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


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    set_determinism(args.seed)
    model, opti = get_model_opti(args)
    args.loss = Loss(loss_type=args.loss, model=model)
    print(args)
    # start training
    TrainHelperMMCBM(args, model, opti).start_train()
