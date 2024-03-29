# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import subprocess
import argparse
import concurrent.futures

from tqdm import tqdm
from inference.init_model import get_name
from params import get_args

# python execute_concept.py -cbm m2 --clip_name cav --cbm_location report_strict -act sigmoid -aow
parser = argparse.ArgumentParser()
parser.add_argument('--wandb', action='store_true', default=False)
parser.add_argument("--name", default='', type=str, help="MRI contrast(default, normal)")
parser.add_argument("--modality", default='MM', type=str, help="MRI contrast(default, normal)")
parser.add_argument('--device', '-d', type=str, default='0')
parser.add_argument('--seed', '-s', type=int, default=32)
parser.add_argument('--concept_shot', '-cs', type=float, default=1.0)
parser.add_argument('--report_shot', '-rs', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--fold', '-f', type=str, default=None)
parser.add_argument('--exclude', '-exc', action='store_true', default=False)
parser.add_argument('--add', action='store_false', default=True)
parser.add_argument('--activation', '-act', default=None, type=str, help='sigmoid, softmax')
parser.add_argument('--backbone', default='SCLS_attnscls_CrossEntropy_32', type=str, help='sigmoid, softmax')
parser.add_argument('--backbone_fold', '-bf', type=int, default=-1)
parser.add_argument('--pos_samples', '-ps', default=50, type=int, help='50, 100')
parser.add_argument('--neg_samples', '-ns', default=0, type=int, help='50, 100')
parser.add_argument('--svm_C', '-C', default=.1, type=float, help='.001, .1')
parser.add_argument('--act_on_weight', '-aow', action='store_true', default=False)
parser.add_argument('--model', type=str, default='b0')
parser.add_argument("--idx", type=int, default=180)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument('--cbm_model', '-cbm', default='mm', type=str, help='mm, s, sa')
parser.add_argument('--fusion', '-fu', type=str, default='max', help='fusion module: max, mean, sum')
parser.add_argument('--init_method', '-im', default='default', type=str)
parser.add_argument('--cbm_location', '-cbl', default='report', type=str, help='params | file | report | human')
parser.add_argument('--clip_name', '-cn', default='cav', type=str,
                    help='clip model name: clip_RN50, RN101, RN50x4, RN50x16, '
                         'RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px'
                         'cav, cav1, cav2')
parser.add_argument('--modality_mask', '-mask', default=False, action='store_true', help='modality mask')
parser.add_argument('--no_data_aug', '-no_aug', action='store_true', default=False)
parser.add_argument('--cav_split', default=0.5, type=float, help='train valid split of cavs')
parser.add_argument('--bias', action='store_true', default=False)
parser.add_argument('--plot_curve', action='store_true', default=False)
parser.add_argument('--weight-norm', action='store_true', default=False)
args = parser.parse_args()
if args.fold:
    args.fold = args.fold.split(',')
    args.fold = [int(i) for i in args.fold]
else:
    args.fold = [0, 1, 2, 3, 4]
print(args.fold)

if isinstance(args.device, str):
    if ',' in args.device:
        args.device = args.device.split(',')
        args.device = [int(i) for i in args.device]
    else:
        args.device = [int(args.device) for _ in range(len(args.fold))]
    print(args.device)
else:
    raise ValueError('device must be None or str')

scripts = 'train_CAV_CBM.py'

name, tail, backbone = get_name(args)
if args.backbone_fold == -1:
    commands = [
        f'python {scripts} --name {name} --backbone {backbone}/fold_{f} --lr {args.lr} --epoch 200 --seed {args.seed} ' \
        f'-k {f} --bz 8 --idx {args.idx} --device {d} --mark r{args.report_shot}_c{args.concept_shot} ' \
        f'-rs {args.report_shot} -cs {args.concept_shot} -act {args.activation} {tail} '
        for f, d in zip(args.fold, args.device)]
else:
    name += f'_backbone_{args.backbone_fold}'
    commands = [
        f'python {scripts} --name {name} --backbone {backbone}/fold_{args.backbone_fold} --lr {args.lr} --epoch 200 --seed {args.seed} ' \
        f'-k {f} --bz 8 --idx {args.idx} --device {d} --mark r{args.report_shot}_c{args.concept_shot} ' \
        f'-rs {args.report_shot} -cs {args.concept_shot} -act {args.activation} {tail}'
        for f, d in zip(args.fold, args.device)]
# Create progress bar
progress_bar = tqdm(total=len(commands), desc='Running Processes')


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    print(f'Starting Process {process.pid}，Executing command：{command}')
    out, err = process.communicate()
    if process.returncode == 0:
        print(f'Process {process.pid} execute successfully')
        # print(f'进程 {process.pid} 的标准输出：')
        # print(out.decode())
    else:
        print(f'Process {process.pid} execute failed')
        print(f'Process {process.pid} Standard error output：')
        print(err.decode())
        print(f'Process {process.pid} Executed command：{command}')
    progress_bar.update(1)


# Use ThreadPoolExecutor to manage subprocesses.
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(run_command, command) for command in commands}

# Close progress bar.
progress_bar.close()
