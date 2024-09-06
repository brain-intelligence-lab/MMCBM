# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""
import os
import subprocess
import argparse
import concurrent.futures
from tqdm import tqdm

# python execute_concept.py -cbm m2 --clip_name cav --cbm_location report_strict -act sigmoid -aow
parser = argparse.ArgumentParser(description="Train CAV CBM model.")

# 添加命令行参数
parser.add_argument('--name', type=str, required=True, help='Name of the experiment')
parser.add_argument('--extra_data', type=str, required=True, help='Extra data description')
parser.add_argument('--device', type=str, required=True, help='GPU device number')
parser.add_argument('--k', type=str, required=True, help='K-fold number')
parser.add_argument('--n_shot', type=str, required=True, help='Mark of the experiment')
parser.add_argument('--epoch', type=int, default=200, help='Mark of the experiment')

args = parser.parse_args()

if args.k:
    args.k = args.k.split(',')
    args.k = [int(i) for i in args.k]
else:
    args.k = [0, 1, 2, 3, 4]
print(args.k)

if isinstance(args.device, str):
    if ',' in args.device:
        args.device = args.device.split(',')
        args.device = [int(i) for i in args.device]
    else:
        args.device = [int(args.device) for _ in range(len(args.k))]
    print(args.device)
else:
    raise ValueError('device must be None or str')

scripts = 'train_RETFound_few_shot.py'
k = args.k
device = args.device
if args.n_shot != 'full':
    args.modality = 'MMOnly'
    args.mark = f'{args.n_shot}shot'
else:
    args.modality = 'MM'
    args.mark = 'full'
commands = []

for f, d in zip(k, device):
    args.k = f
    args.device = d
    commands.append(
        ' '.join([f'python {scripts}'] + [f'--{k} {v}' for k, v in vars(args).items()])
    )

# Create progress bar
progress_bar = tqdm(total=len(commands), desc='Running Processes')


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    print(f'Starting Process {process.pid}，Executing command：{command}')
    out, err = process.communicate()
    if process.returncode == 0:
        print(f'Process {process.pid} execute successfully')
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
