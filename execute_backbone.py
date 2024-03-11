import subprocess
import argparse
import concurrent.futures

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--device', '-d', type=str, default=None)
parser.add_argument('--seed', '-s', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--idx', type=int, default=180)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--fold', '-f', type=str, default=None)
parser.add_argument('--fusion', '-fu', type=str, default='pool')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--model', type=str, default='b0')
parser.add_argument("--modality", default='MM', type=str, help="MRI contrast(default, normal)")

args = parser.parse_args()

if args.fold is None:
    args.fold = [0, 1, 2, 3, 4]
elif isinstance(args.fold, str):
    args.fold = args.fold.split(',')
    args.fold = [int(i) for i in args.fold]
else:
    raise ValueError('fold must be None or str')

if args.device is None:
    args.device = [0, 1, 2, 3, 4]
elif isinstance(args.device, str):
    args.device = args.device.split(',')
    args.device = [int(i) for i in args.device]
else:
    raise ValueError('device must be None or str')

scripts = 'train_efficient_scls.py'
name = f'attnscls_CrossEntropy_{args.seed}'

tail = ''
if args.fusion != 'pool':
    name += f'_{args.fusion}'
    tail += f' --fusion {args.fusion}'

if args.model != 'b0':
    tail += f' --model {args.model}'

if args.modality != 'MM':
    name += f'_{args.modality}'
    tail += f' --modality {args.modality}'

if args.infer:
    tail += ' --infer'

if args.resume:
    tail += ' --resume'

commands = [
    f'python {scripts} --name {name} --lr {args.lr} --epochs {args.epochs} --seed {args.seed} ' \
    f'-k {f} --bz 8 --idx {args.idx} --device {d} {tail}'
    for d, f in zip(args.device, args.fold)]
# 创建进度条
progress_bar = tqdm(total=len(commands), desc='Running Processes')


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    print(f'启动进程 {process.pid}，执行命令：{command}')
    out, err = process.communicate()
    if process.returncode == 0:
        print(f'进程 {process.pid} 执行成功')
        # print(f'进程 {process.pid} 的标准输出：')
        # print(out.decode())
    else:
        print(f'进程 {process.pid} 执行失败')
        print(f'进程 {process.pid} 的标准错误输出：')
        print(err.decode())
        print(f'进程 {process.pid} 执行的命令：{command}')
    progress_bar.update(1)


# 使用 ThreadPoolExecutor 管理子进程
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(run_command, command) for command in commands}

# 关闭进度条
progress_bar.close()
