import os

import torch

from pathlib import Path
from utils.decorator import decorator_args
from trainer.train_utils import EarlyStopping
from utils.logger import JsonLogs, char_color, make_dirs
from monai.utils import set_determinism


def get_name(args):
    name = f'{args.name}{args.cbm_model}CBM'
    tail = f' --cbm_model {args.cbm_model}'

    if args.clip_name != 'cav':
        name += f'_{args.clip_name}'
        tail += f' --clip_name {args.clip_name}'

    name += f'_{args.activation}'
    tail += f' --activation {args.activation}'

    name += f'_C{args.svm_C}'
    tail += f' --svm_C {args.svm_C}'

    name += f'CrossEntropy{args.name}_{args.seed}'
    backbone = f'Efficient{args.model}_{args.backbone}'

    name += f'_{args.cbm_location}'
    tail += f' --cbm_location {args.cbm_location}'

    if args.add:
        name += '_add'
        tail += ' --add'

    if args.pos_samples != 50:
        name += f'_n{args.pos_samples}'
        tail += f' --pos_samples {args.pos_samples}'
    if args.neg_samples != 0:
        name += f'_neg{args.neg_samples}'
        tail += f' --neg_samples {args.neg_samples}'

    if args.cav_split != 0.5:
        name += f'_cav{args.cav_split}'
        tail += f' --cav_split {args.cav_split}'

    if args.act_on_weight:
        name += '_aow'
        tail += ' --act_on_weight'

    if args.model != 'b0':
        name += f'_{args.model}'
        tail += f' --model {args.model}'

    if args.init_method != 'default':
        name += f'_{args.init_method}'
        tail += f' --init_method {args.init_method}'

    if args.epochs != 200 and args.epochs != 0:
        name += f'_e{args.epochs}'
        tail += f' --epochs {args.epochs}'

    if args.modality_mask:
        name += '_mask'
        tail += ' --modality_mask'

    if args.no_data_aug:
        name += '_no_aug'
        tail += ' --no_data_aug'

    if args.bias:
        name += '_bias'
        tail += ' --bias'

    if args.plot_curve:
        tail += ' --plot_curve'

    if args.weight_norm:
        name += '_weight_norm'
        tail += ' --weight_norm'

    name += f'_{args.modality}'
    tail += f' --modality {args.modality}'

    name += f'_{args.fusion}'
    tail += f' --fusion {args.fusion}'
    return name, tail, backbone


def init_model(k, cs=1, seed=32, cbm_location='report', cbm_model='m2', device='cpu', clip_name='cav',
               activation='sigmoid', pos_samples=50, neg_samples=0, resume_epoch=180, cav_split=.5,
               backbone='SCLS_attnscls_CrossEntropy_32_add', bias=False, init_method='zero', act_on_weight=False,
               json_path=None, ignore_keys=('device',)):
    @decorator_args
    def get_args(args):
        # enabling cudnn determinism appears to speed up training by a lot
        args.down_sample = False
        args.prognosis = False
        # save_name = 'Model'
        args.bias = bias
        args.init_method = init_method
        args.clip_name = clip_name
        args.pos_samples = pos_samples
        args.neg_samples = neg_samples
        args.cav_split = cav_split
        args.act_on_weight = act_on_weight

        ##################### debug #####################
        args.device = device
        args.k = k
        args.epochs = 0
        args.num_worker = 0
        args.infer = True
        args.concept_shot = float(cs)
        args.modality = 'MM'
        args.cbm_model = cbm_model
        args.fusion = 'max'
        args.cbm_location = cbm_location
        args.seed = seed

        args.mark = f'r1.0_c{args.concept_shot}'
        args.mark = f'{args.cbm_location}_{args.mark}'
        args.infer = False
        args.resume = True
        args.activation = activation
        args.idx = 180
        args.backbone = f'Efficientb0_{backbone}/fold_{args.k}'
        args.add = True
        args.analysis_top_k = None
        args.analysis_threshold = None
        ##################### debug #####################
        args.name = get_name(args)[0]
        if 'clip' in args.clip_name:
            args.dir_name = f'CLip'
        else:
            args.dir_name = f'CAV'

        args.kwargs = {}
        print(args)
        args.metrics = []
        args.mode = 'max'

    args = get_args()
    # load saved configs
    if json_path is not None:
        fold_name = os.path.basename(json_path).split('_')
        fold_name[1] = k
        fold_name = '_'.join([str(i) for i in fold_name])
        json_path = os.path.join(os.path.dirname(json_path), fold_name)
        file_name = 'args.json'
        if not os.path.exists(json_path):
            print(char_color(f'json path {json_path} not exists, use the txt file instead', color='red'))
            file_name = sorted([i for i in os.listdir(json_path) if 'param' in i])[-1]
        args = JsonLogs(dir_path=json_path, file_name=file_name).read(args, ignore_keys=ignore_keys)
    # 加载模型
    set_determinism(args.seed)
    if args.clip_name == 'backbone':
        from train_efficient_scls import get_model_opti
    else:
        from train_CAV_CBM import get_model_opti
    model, opti = get_model_opti(args)
    print(args.dir_name)
    stoppers = EarlyStopping(dir=f'{args.output_dir}/{args.dir_name}/epoch_stopper',
                             patience=args.patience,
                             mode=args.mode)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    make_dirs(f'{args.output_dir}/{args.dir_name}')
    print(char_color(f'Using device {args.device}'))
    print(char_color(f'Using path {args.output_dir}/{args.dir_name}'))
    stoppers.load_checkpoint(model=model, ignore=args.ignore,
                             name=f'checkpoint_{resume_epoch}.pth' if args.idx else 'MM.pth')
    model.to(device=args.device)
    model.eval()
    return model, args


def init_model_json(json_path, k=None, resume_epoch=180, **kwargs):
    @decorator_args
    def get_args(args):
        # enabling cudnn determinism appears to speed up training by a lot
        for k, v in kwargs.items():
            setattr(args, k, v)

    json_path = Path(json_path)
    # update kwargs to ignore_keys
    kwargs.update({'idx': 180, 'output_dir': json_path.parents[-2]})
    ignore_keys = list(kwargs.keys())

    args = get_args()
    # load saved configs
    if k is not None:
        fold_name = json_path.name.split('_')
        fold_name[1] = k
        fold_name = '_'.join([str(i) for i in fold_name])
        json_path = json_path.parent.joinpath(fold_name)
    file_name = 'args.json'
    if not json_path.joinpath('TXTLogger', file_name).exists():
        print(char_color(f'json path {json_path} not exists, use the txt file instead', color='red'))
        file_name = sorted([i.name for i in json_path.joinpath('TXTLogger').iterdir() if 'param' in i.name])[-1]
    args = JsonLogs(dir_path=json_path, file_name=file_name).read(args, ignore_keys=ignore_keys)
    # 加载模型
    set_determinism(args.seed)
    if args.clip_name == 'backbone':
        from train_efficient_scls import get_model_opti
    else:
        from train_CAV_CBM import get_model_opti

    model, opti = get_model_opti(args)
    print(args.dir_name)
    stoppers = EarlyStopping(dir=f'{args.output_dir}/{args.dir_name}/epoch_stopper',
                             patience=args.patience,
                             mode=args.mode)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    make_dirs(f'{args.output_dir}/{args.dir_name}')
    print(char_color(f'Using device {args.device}'))
    print(char_color(f'Using path {args.output_dir}/{args.dir_name}'))
    stoppers.load_checkpoint(model=model, ignore=args.ignore,
                             name=f'checkpoint_{resume_epoch}.pth' if args.idx else 'MM.pth')
    model.to(device=args.device)
    model.eval()
    return model, args
