# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import os
import time

import torch
import numpy as np
from .logger import PrintColor


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, dir='output', name='checkpoint', best_score=None, patience=7, verbose=False, delta=0,
                 mode='max', device='cpu', logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode ('max'|'min'): 'max' means the larger the evaluation indicator, the better
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.name = name
        self.dir = dir
        self.mode = mode
        self.device = device
        self.logger = logger
        try:
            os.makedirs(self.dir, exist_ok=True)
            os.chmod(self.dir, mode=0o777)
            print(f'{dir} created')
        except Exception:
            print(f"{dir} :EarlyStop make dir Error")

    def char_color(self, s, front=50, word=32):
        """
        # 改变字符串颜色的函数
        :param s:
        :param front:
        :param word:
        :return:
        """
        new_char = "\033[0;" + str(int(word)) + ";" + str(int(front)) + "m" + s + "\033[0m"
        return new_char

    def __call__(self, val_record, model, optimizer=None, whole=False, epoch=None, step=10):
        score = val_record
        if whole:
            if epoch % step == 0:
                self.save_checkpoint(val_record, model, optimizer, epoch)
            return

        if self.best_score == torch.nan:
            print(f'{self.name}: there is nan in score')
            raise ValueError

        if self.best_score is None:
            self.best_score = score
            print("Early stop initiated")
            self.save_checkpoint(val_record, model, optimizer)
            return

        if self.mode == 'max':
            if score <= self.best_score + self.delta:
                self.counter += 1
                PrintColor(
                    f'{self.name} EarlyStopping counter: {self.counter} out of {self.patience}       best score: {self.best_score}'
                )

                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                PrintColor(f"{self.name} {self.best_score} -> {score}")
                self.best_score = score
                self.save_checkpoint(val_record, model, optimizer)
                self.counter = 0
        elif self.mode == 'min':
            if score >= self.best_score + self.delta:
                self.counter += 1
                PrintColor(
                    f'{self.name} EarlyStopping counter: {self.counter} out of {self.patience}       best score: {self.best_score}'
                )

                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                PrintColor(f"{self.name} {self.best_score} -> {score}")
                self.best_score = score
                self.save_checkpoint(val_record, model, optimizer)
                self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer=None, epoch=None):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if epoch is None:
            name = self.name + '.pth'
        else:
            name = self.name + f'_{epoch}.pth'
        if self.logger:
            self.logger({
                "name": self.name,
                "best_loss": self.best_score,
                "best_score": self.best_score,
            })
        if self.verbose:
            PrintColor(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if optimizer is None:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_loss": self.best_score,
                    "best_score": self.best_score,
                },
                f"{os.path.join(self.dir, name)}"
            )
        else:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_loss": self.best_score,
                    "best_score": self.best_score,
                    "optimizer": dict([optimizer.state_dict()]) if hasattr(optimizer, 'state_dict') else None,
                    'lr': optimizer.param_groups[0]['lr'] if optimizer is not None else None,
                },
                f"{os.path.join(self.dir, name)}"
            )
        self.val_loss_min = val_loss

    def load_checkpoint(self, model, ignore=False, cp=True, name=None, strict=False):
        """
        :param model:
        :param ignore:
        :param cp: save a copy while load a checkpoint
        :return:
        """
        if name is None:
            name = self.name + '.pth'
        if not os.path.exists(f'{self.dir}/{name}'):
            dir_path = os.path.dirname(self.dir)
        else:
            dir_path = self.dir
        try:
            checkpoint = torch.load(f'{dir_path}/{name}', map_location=self.device)
            if cp or not ignore:
                os.system(
                    f'cp {dir_path}/{name} {dir_path}/{name}_{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}'
                )
        except FileNotFoundError:
            raise FileNotFoundError(f"{dir_path}/{name} NOT EXISTS")

        try:
            model.load_state_dict(checkpoint["state_dict"], strict=strict)
            if not ignore:
                self.best_score = checkpoint["best_score"]
        except KeyError:
            model.load_state_dict(checkpoint, strict=strict)
        except RuntimeError:
            raise RuntimeError(f'\033[1;31mmodel[{name}] load failed\033[0m')

        PrintColor(f"{name} load score: {self.best_score} from {dir_path}")
        model.eval()
        return True
