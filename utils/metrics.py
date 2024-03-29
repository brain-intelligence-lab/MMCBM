# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import torch
import re
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
    roc_auc_score, confusion_matrix
from torchmetrics.retrieval import RetrievalRecall, RetrievalPrecision
from functools import wraps


class BaseObject(nn.Module):
    def __init__(self, name=None, average=None, gross=False):
        super().__init__()
        self.average = average
        self.gross = gross
        self._name = name if name is not None else re.sub('([a-z0-9])([A-Z])', r'\1_\2',
                                                          re.sub('(.)([A-Z][a-z]+)', r'\1_\2',
                                                                 self.__class__.__name__)).lower()
        if average is not None:
            self._name += f'_{average}'

    @property
    def __name__(self):
        return self._name

    def activation(self, pr, num_classes):
        pr = torch.softmax(torch.tensor(pr[..., :num_classes]), dim=1)
        return pr

    def to_pred(self, pr):
        return pr.argmax(-1)

    def to_onehot(self, pred, num_classes):
        pred = pred.reshape(-1, 1)
        return torch.zeros(pred.shape[0], num_classes).scatter_(1, pred, 1)

    def compute(self, y_label, y_label_onehot, y_pred, y_pred_onehot, y_score):
        pass

    def forward(self, pre, y_label, stage_name):
        num_classes = 3
        if stage_name == 'train':
            num_classes = pre.shape[-1]
        y_score = self.activation(pre, num_classes)
        y_pred = self.to_pred(y_score)
        y_pred_onehot = self.to_onehot(y_pred, num_classes)
        if isinstance(y_label, (list, tuple)):
            y_label = y_label[-1]
        y_label_onehot = self.to_onehot(y_label, num_classes)
        return self.compute(y_label, y_label_onehot, y_pred, y_pred_onehot, y_score)


class Accuracy(BaseObject):
    def __init__(self):
        super().__init__()

    def compute(self, y_label, y_label_onehot, y_pred, y_pred_onehot, y_score):
        return accuracy_score(
            y_label, y_pred,
        )


class Recall(BaseObject):
    def __init__(self, average='macro'):
        super().__init__(average=average)

    def compute(self, y_label, y_label_onehot, y_pred, y_pred_onehot, y_score):
        return recall_score(
            y_label, y_pred, average=self.average
        )


class Precision(BaseObject):
    def __init__(self, average='macro'):
        super().__init__(average=average)

    def compute(self, y_label, y_label_onehot, y_pred, y_pred_onehot, y_score):
        return precision_score(
            y_label, y_pred, average=self.average
        )


class F1(BaseObject):
    def __init__(self, average='macro'):
        super(F1, self).__init__(average=average)

    def compute(self, y_label, y_label_onehot, y_pred, y_pred_onehot, y_score):
        return f1_score(
            y_label, y_pred, average=self.average
        )


class AUC(BaseObject):
    def __init__(self, average='macro'):
        super(AUC, self).__init__(average=average, gross=True)

    def compute(self, y_label, y_label_onehot, y_pred, y_pred_onehot, y_score):
        return roc_auc_score(
            y_label_onehot, y_score, average=self.average, multi_class='ovo'
        )


class ConfusionMatrix(BaseObject):
    def __init__(self, average='macro'):
        super(ConfusionMatrix, self).__init__(average=average, gross=True)

    def compute(self, y_label, y_label_onehot, y_pred, y_pred_onehot, y_score):
        return confusion_matrix(
            y_label, y_pred
        )


# ----------------------------------- retrieval ----------------------------------- #
# retrieval metrics
# ----------------------------------- retrieval ----------------------------------- #
def retrieval_patient(function):
    @wraps(function)
    def decorated(gt, pre, k, search_num, modality=None, mean=True):
        if modality is not None:
            gt = gt[gt['modality'] == modality]
            pre = pre[pre['modality'] == modality]
            pre = pre.groupby(['name']).apply(
                lambda x: x.sort_values(by='score', ascending=False)[:search_num]
            ).reset_index(drop=True)
        else:
            pre = pre.groupby(['name', 'modality']).apply(
                lambda x: x.sort_values(by='score', ascending=False)[
                          :int(search_num * {'FA': 0.3, 'ICGA': 0.2, 'US': 0.5}[x['modality'].iloc[0]])]).reset_index(
                drop=True)
        metrics = []
        for name in set(gt['name'].unique()) & set(pre['name'].unique()):
            # name_gt = gt[gt['name'] == name][['concept', 'modality', 'time']].drop_duplicates()
            # name_mm = pre[pre['name'] == name][['concept', 'modality', 'score']].drop_duplicates()

            name_gt = gt[gt['name'] == name][['concept', 'time', 'modality']].drop_duplicates()
            name_mm = pre[pre['name'] == name][['concept', 'time', 'modality', 'score']].drop_duplicates()
            name_gt['concept'] = name_gt['time'] + name_gt['concept']
            name_mm['concept'] = name_mm['time'] + name_mm['concept']
            preds = torch.tensor(name_mm['score'].to_list())
            target = torch.tensor(name_mm['concept'].isin(name_gt['concept']).to_list())
            metrics.append(function(preds, target, k))
        return np.mean(metrics) if mean else metrics

    return decorated


@retrieval_patient
def retrieval_precision_patient(preds, target, k):
    rp = RetrievalPrecision(empty_target_action='neg', top_k=k)
    return rp(preds, target, indexes=torch.ones(len(preds), dtype=torch.long))


@retrieval_patient
def retrieval_recall_patient(preds, target, k):
    rr = RetrievalRecall(empty_target_action='neg', top_k=k)
    return rr(preds, target, indexes=torch.ones(len(preds), dtype=torch.long))


@retrieval_patient
def retrieval_f1_patient(preds, target, k):
    precision = RetrievalPrecision(empty_target_action='neg', top_k=k)(
        preds, target, indexes=torch.ones(len(preds), dtype=torch.long))
    recall = RetrievalRecall(empty_target_action='neg', top_k=k)(
        preds, target, indexes=torch.ones(len(preds), dtype=torch.long))
    return 2 * precision * recall / (precision + recall + 1e-7)


def retrieval_ranks(gt, pre, k, modality=None):
    if modality is not None:
        gt = gt[gt['modality'] == modality]
        pre = pre[pre['modality'] == modality]
    metrics = []
    for name in set(gt['name'].unique()) & set(pre['name'].unique()):
        name_gt = gt[gt['name'] == name][['concept', 'modality']].drop_duplicates()
        predictions = pre[pre['name'] == name][['concept', 'modality', 'score']].drop_duplicates().sort_values(
            by='score', ascending=False)
        ranks_k = [i + 1 for i, item in enumerate(predictions['concept'][:k]) if item in name_gt['concept'].to_list()]
        # Median rank
        if ranks_k:
            median_rank = sorted(ranks_k)[len(ranks_k) // 2]
            # Mean rank
            mean_rank = sum(ranks_k) / len(ranks_k)

            # Mean reciprocal rank
            mrr = 1 / ranks_k[0]
            metrics.append({
                'median rank': median_rank,
                'mean rank': mean_rank,
                'mean reciprocal rank': mrr
            })
    return pd.DataFrame(metrics).mean().to_dict()
