# third-party imports
import os
import torch
import pandas as pd
import torch.optim.lr_scheduler
import numpy as np
# project imports
from utils.logger import char_color, make_dirs, Logs, CSVLogs, JsonLogs
from utils.dataloader import get_loaders_from_args
from utils.EarlyStop import EarlyStopping
import warnings
from visualize.utils import plot_curve, plot_confusion_matrix

warnings.filterwarnings("ignore")


class AverageValueMeter(object):
    """
    Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        """
        Log a new value to the meter
        @param value: Next result to include.
        @param n: number
        """
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = 0, 0
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        """
        Get the value of the meter in the current state.
        """
        return self.mean, self.std

    def reset(self):
        """
        Resets the  meter to default settings.
        """
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = 0.0
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = 0.0


class PolyLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, epochs):
        super().__init__(optimizer, lambda epoch: (1 - epoch / epochs) ** 0.9)
        self.epochs = epochs


class PolyOptimizer:
    def __init__(self, optimizer, epochs):
        self.optimizer = optimizer
        self.lr_scheduler = PolyLR(optimizer, epochs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def lr_step(self):
        self.lr_scheduler.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


def train_fold(args, model, optimizer, epocher_fn, infer_fn=None):
    """
    :param args:
    :param model:
    :param optimizer:
    """
    if args.wandb:
        import wandb
        wandb.init(project=args.dir_name, reinit=True,
                   config=args, name='_'.join(args.dir_name.split('/')[1:]))
        wandb.watch(model, args.loss, log='all', log_freq=10)

    JsonLogs(dir_path=f'{args.output_dir}/{args.dir_name}')(args)
    stoppers = EarlyStopping(dir=f'{args.output_dir}/{args.dir_name}/epoch_stopper',
                             patience=args.patience,
                             mode=args.mode)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    make_dirs(f'{args.output_dir}/{args.dir_name}')
    print(char_color(f'Using device {args.device}'))
    print(char_color(f'Using path {args.output_dir}/{args.dir_name}'))
    if args.resume:
        stoppers.load_checkpoint(model=model, ignore=args.ignore,
                                 name=f'checkpoint_{args.idx}.pth' if args.idx else 'MM.pth')
    model.to(device=args.device)

    loaders = get_loaders_from_args(args)
    if args.infer:
        return None if infer_fn is None else infer_fn(args, model, loaders)

    # output parameters
    pred_logger = CSVLogs(dir_path=f'{args.output_dir}/{args.dir_name}', file_name=f'pred_output')
    metrics_logger = CSVLogs(dir_path=f'{args.output_dir}/{args.dir_name}', file_name=f'metrics_output')

    if args.epochs > 0 and not args.plot_curve:
        logger = Logs(dir_path=f'{args.output_dir}/{args.dir_name}', file_name='param')

        for key, value in args.__dict__.items():
            logger.log(f'{key}: {value}')
        logger = Logs(dir_path=f'{args.output_dir}/{args.dir_name}', file_name='print')
        optimizer = PolyOptimizer(optimizer, args.epochs)
        epochers = epocher_fn(args, model, optimizer, logger, pred_logger, metrics_logger)
        train(args, model, loaders, epochers, optimizer, stoppers, logger)
        pred_logger.output_error_name()

    if args.plot_curve or args.epochs > 0:
        save_metrics(args)
        plot_curve(f'{args.output_dir}/{args.dir_name}', metrics_logger, hue='modality', split='stage_name')
        plot_curve(f'{args.output_dir}/{args.dir_name}', metrics_logger, hue='stage_name', split='modality')
        plot_confusion_matrix(f'{args.output_dir}/{args.dir_name}', csv_logger=pred_logger,
                              stage_name='valid', prognosis=args.prognosis)
        plot_confusion_matrix(f'{args.output_dir}/{args.dir_name}', csv_logger=pred_logger,
                              stage_name='test', prognosis=args.prognosis)
        # plot_roc_curve(f'{args.output_dir}/{args.dir_name}', pred_logger)

    del model
    if args.wandb:
        import wandb
        wandb.finish()


def train(args, model, loaders, epochers, optimizer, stoppers, logger):
    train_loader, val_loader, test_loader = loaders
    trainepoch, validepoch, testepoch = epochers
    if args.plot:
        print('plotting')
        if not train_loader.is_empty():
            trainepoch.plot_epoch(train_loader)
        if not val_loader.empty():
            validepoch.plot_epoch(val_loader)
        if not test_loader.empty():
            testepoch.plot_epoch(test_loader)
        raise SystemExit

    for i in range(args.epochs):
        # record learning rate
        print(char_color(f" [Epoch: {i}/{args.epochs}], "
                         f"lr: {optimizer.param_groups[0]['lr']}, "
                         f"path: {args.dir_name}"))

        logger.log(f"[Epoch: {i}/{args.epochs}], "
                   f"lr: {optimizer.param_groups[0]['lr']}, "
                   f"path: {args.dir_name}")

        # start epoch
        trainlogs, trainbest = trainepoch(train_loader)
        logger.log(f'train: {trainlogs}')
        logger.log(f'train_best: {trainbest}')

        if not val_loader.empty():
            validlogs, validbest = validepoch(val_loader)
            logger.log(f'val: {validlogs}')
            logger.log(f'val_best: {validbest}')

        if not test_loader.empty():
            testlogs, testbest = testepoch(test_loader)
            logger.log(f'test: {testlogs}')
            logger.log(f'test_best: {testbest}')

        # early stopper update
        stoppers(
            val_record=None,
            model=model,
            whole=True,
            epoch=i,
            step=10
        )


def save_metrics(args):
    dir_path, epochs = f'{args.output_dir}/{args.dir_name}', args.epochs
    get_stage1_metrics_from_top_stage2(path=dir_path,
                                       stage1='valid',
                                       top_k=1)
    get_stage1_metrics_from_top_stage2(path=dir_path,
                                       stage1='test',
                                       stage2='valid',
                                       top_k=3)

    get_stage1_metrics_from_top_stage2(path=dir_path,
                                       stage1='test',
                                       stage2='valid',
                                       top_k=2)

    get_stage1_metrics_from_top_stage2(path=dir_path,
                                       stage1='test',
                                       stage2='valid',
                                       top_k=1)
    if epochs is not None:
        get_stage1_metrics_from_top_stage2(path=f'{dir_path}',
                                           stage1='test',
                                           top_k=1,
                                           epoch=epochs)


def plot_metrics_from_csv(dir_path):
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        logger = CSVLogs(dir_path=path, file_name=f'metrics_output')
        plot_curve(path, logger)


def top_metrics_of_stage(df, stage, modality, metric, k=1, epoch=None):
    if epoch is None:
        df_s = df[(df['stage_name'] == stage) & (df['modality'] == modality)].sort_values(
            by=metric, ascending=False).iloc[:k]
    else:
        df_s = df[(df['stage_name'] == stage) & (df['modality'] == modality) & (df['epoch'] == epoch)]
    return df_s


def load_metrics(path,  file_name='metrics_output'):
    if isinstance(path, pd.DataFrame):
        df = path
    else:
        logger = CSVLogs(dir_path=path, file_name=file_name)
        df = logger.dataframe()
    metrics = [c for c in df.columns if 'gross' in c]
    df = df[['stage_name', 'epoch', 'modality', *metrics]]
    return df


def get_stage1_metrics_from_top_stage2(path, stage1, stage2=None, top_k=1, epoch=None, mean=True, main_modality=None,
                                       save=True, file_name='metrics_output'):
    df = load_metrics(path, file_name)
    if df[df['stage_name'] == stage1].empty:
        return
    metrics = [c for c in df.columns if 'gross' in c]
    best_metric = []
    if main_modality is not None and main_modality not in df['modality'].unique():
        raise Exception(f'{main_modality} not in {df["modality"].unique()}')

    if main_modality is not None:
        for metric in metrics:
            if stage2:
                df_s = top_metrics_of_stage(df, stage2, main_modality, metric, top_k)
            else:
                df_s = top_metrics_of_stage(df, stage1, main_modality, metric, top_k, epoch)
            df_s = pd.merge(df[df['stage_name'] == stage1], df_s['epoch'], on='epoch', how='inner').copy()
            df_s['select'] = metric
            best_metric.append(df_s)
    else:
        for m in df['modality'].drop_duplicates():
            for metric in metrics:
                if stage2:
                    df_s = top_metrics_of_stage(df, stage2, m, metric, top_k)
                    df_s = pd.merge(df[(df['stage_name'] == stage1) & (df['modality'] == m)],
                                    df_s['epoch'], on='epoch', how='inner').copy()
                else:
                    df_s = top_metrics_of_stage(df, stage1, m, metric, top_k, epoch)
                df_s['select'] = metric
                best_metric.append(df_s)
    best_metric = pd.concat(best_metric)
    if mean:
        try:
            best_metric = best_metric.groupby(['stage_name', 'select', 'modality']).mean()
        except Exception:
            import pdb
            pdb.set_trace()
    if not save:
        return best_metric.reset_index()
    if epoch is not None:
        best_metric.to_csv(f'{path}/{stage1}_best_top{top_k}_epoch{epoch}.csv')
        return
    best_metric.to_csv(f'{path}/{stage1}_best_top{top_k}.csv')


def read_csv_scores(dir_path, epoch=None, mean=False, main_modality=None, read_csv=False):
    df = []
    for sub in os.listdir(dir_path):
        path = os.path.join(dir_path, sub)
        if not os.path.exists(os.path.join(path, 'valid_best_top1.csv')):
            print(sub)
            continue

        if epoch is not None:
            if os.path.exists(os.path.join(path, 'CSVLogger', f'metrics_output_{epoch}.csv')):
                valid = pd.read_csv(os.path.join(path, 'CSVLogger', f'metrics_output_{epoch}.csv'))
                test = pd.DataFrame()
            else:
                test = get_stage1_metrics_from_top_stage2(path=path,
                                                          stage1='test',
                                                          top_k=1,
                                                          epoch=epoch,
                                                          save=False)
                valid = get_stage1_metrics_from_top_stage2(path=path,
                                                           stage1='valid',
                                                           top_k=1,
                                                           epoch=epoch,
                                                           save=False)
        else:
            if read_csv:
                valid = pd.read_csv(os.path.join(path, f'valid_best_top1.csv'))
                test = pd.read_csv(os.path.join(path, f'test_best_top1.csv'))
            else:
                valid = get_stage1_metrics_from_top_stage2(path=path,
                                                           stage1='valid',
                                                           top_k=1,
                                                           save=False,
                                                           main_modality=main_modality)
                test = get_stage1_metrics_from_top_stage2(path=path,
                                                          stage1='test',
                                                          stage2='valid',
                                                          top_k=1,
                                                          save=False,
                                                          main_modality=main_modality)

        data = pd.concat([valid, test])
        if 'select' in data.columns:
            data = data[data['select'] == 'gross_f1_macro'][
                ['stage_name', 'modality', 'epoch', 'gross_accuracy', 'gross_precision_macro', 'gross_recall_macro',
                 'gross_f1_macro']]
        else:
            data = data[['stage_name', 'modality', 'gross_accuracy', 'gross_precision_macro', 'gross_recall_macro',
                         'gross_f1_macro']]
        data['fold'] = float(sub.split('_')[1])
        try:
            checkpoint = torch.load(f'{path}/MM.pth', map_location='cpu')['state_dict']

            def get_concept_num(df):
                modality = df['modality'].unique()[0]
                if len(checkpoint) == 1:
                    df['cn'] = int(list(checkpoint.values())[0].shape[1])
                elif 'MM' in modality:
                    df['cn'] = sum([int(v.shape[1]) for v in checkpoint.values()])
                else:
                    for k, v in checkpoint.items():
                        if modality in k:
                            df['cn'] = int(v.shape[1])
                return df

            data = data.groupby('modality').apply(get_concept_num).reset_index(drop=True)
        except:
            data['cn'] = 0
        try:
            data['rs'] = float(sub.split('_')[-2][1:])
            data['cs'] = float(sub.split('_')[-1][1:])
        except:
            print(f'Error: {sub}')
            data['rs'] = 1.0
            data['cs'] = 1.0

        try:
            data['rn'] = int(float(sub.split('_')[-2][1:]) * 97)
        except:
            print(f'Rn Error: {sub}')
            data['rn'] = 1.0
        df.append(data)
    df = pd.concat(df)
    # df = df.drop('fold', axis=1)
    df = df.rename(columns={'gross_accuracy': 'accuracy', 'gross_precision_macro': 'precision',
                            'gross_recall_macro': 'recall', 'gross_f1_macro': 'f1 score'})
    if mean:
        return df.groupby(['rs', 'cs', 'cn', 'modality', 'stage_name'], axis=0).mean().reset_index()
    df = df.drop(columns=['fold', 'epoch'], axis=1)
    return df[(df['modality'] == 'MM') | (df['modality'] == 'FA') | (df['modality'] == 'ICGA') | (
            df['modality'] == 'US')]