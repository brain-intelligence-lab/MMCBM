import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLoss


class MSE(BaseLoss):
    def __init__(self, reduction="mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.criterion = nn.MSELoss(reduction=reduction)

    def compute(self, pred, inp):
        loss = self.criterion(input=pred, target=inp)  # (Num_nodes, 4)
        return loss


class CrossEntropy(BaseLoss):
    def __init__(self, model=None, **kwargs):
        super(CrossEntropy, self).__init__(model=model, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def compute(self, pre, inp):
        loss = self.loss(pre, inp)
        if self.model is not None:
            loss += torch.linalg.vector_norm(self.model.weight, ord=1, dim=-1).max() * 1e-2
        return loss


class CrossFocal(BaseLoss):
    """
     One possible pytorch implementation of focal loss (https://arxiv.org/abs/1708.02002), for multiclass classification.
     This module is intended to be easily swappable with nn.CrossEntropyLoss.
     If with_logits is true, then input is expected to be a tensor of raw logits, and a softmax is applied
     If with_logits is false, then input is expected to be a tensor of probabiltiies
     target is expected to be a batch of integer targets (i.e. sparse, not one-hot). This is the same behavior as
     nn.CrossEntropyLoss.
     This loss also ignores contributions where target == ignore_index, in the same way as nn.CrossEntropyLoss
     batch behaviour: reduction = 'none', 'mean', 'sum'
     """

    def __init__(self, gamma=2, eps=1e-7, ignore_index=-100, reduction='mean', label_smoothing=0,
                 class_weight=(1, 1, 1, 1), **kwargs):
        super().__init__(
            class_weight=class_weight,
            ignore_index=ignore_index,
            gamma=gamma,
            eps=eps,
            reduction=reduction,
            label_smoothing=label_smoothing,
            **kwargs
        )
        print(
            f'Focal weight: {class_weight}, gamma: {gamma}, eps: {eps}, ignore_index: {ignore_index}, reduction: {reduction}')
        self.alpha = torch.tensor(class_weight, dtype=torch.float32)

    def compute(self, input, target):
        """
        A function version of focal loss, meant to be easily swappable with F.cross_entropy. The equation implemented here
        is L_{focal} = - \sum (1 - p_{target})^\gamma p_{target} \log p_{pred}
        If with_logits is true, then input is expected to be a tensor of raw logits, and a softmax is applied
        If with_logits is false, then input is expected to be a tensor of probabiltiies (softmax previously applied)
        target is expected to be a batch of integer targets (i.e. sparse, not one-hot). This is the same behavior as
        nn.CrossEntropyLoss.
        Loss is ignored at indices where the target is equal to ignore_index
        batch behaviour: reduction = 'none', 'mean', 'sum'
        """
        y = F.one_hot(target, input.size(-1))
        if self.label_smoothing > 0:
            c = y.size(-1)
            y = y * (1 - self.label_smoothing) + self.label_smoothing / c
        pt = F.softmax(input, dim=-1) + self.eps  # avoid nan

        cross_entropy = -y * torch.log(pt)  # cross entropy
        loss = cross_entropy * (1 - pt) ** self.gamma  # focal loss factor
        alpha_weight = self.alpha.to(loss.device)[target.squeeze().long()].view(-1, 1)
        loss = loss * alpha_weight
        loss = torch.sum(loss, dim=-1)

        # batch reduction
        if self.reduction == 'mean':
            return torch.mean(loss, dim=-1)
        elif self.reduction == 'sum':
            return torch.sum(loss, dim=-1)
        else:  # 'none'
            return loss


class DiceLoss(BaseLoss):
    """
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        output: A tensor of shape [N, C, *]
        target: A tensor of same shape with output
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, class_weight=None, ignore_index=None, batch_dice=False, label_smoothing=0, **kwargs):
        super(DiceLoss, self).__init__(class_weight=torch.tensor(class_weight, dtype=torch.float32),
                                       label_smoothing=label_smoothing,
                                       batch_dice=batch_dice, reduction='mean', smooth=1, **kwargs)
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

    def compute(self, output, target):
        weight = self.class_weight.to(output.device)[target.squeeze().long()].view(-1, 1)

        if output.shape != target.shape:
            target = F.one_hot(target, output.size(-1))

        if self.label_smoothing > 0:
            c = target.size(-1)
            target = target * (1 - self.label_smoothing) + self.label_smoothing / c
        output = F.softmax(output, dim=1)

        output = output.contiguous().view(output.shape[0], -1)
        target = target.contiguous().view(output.shape[0], -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=-1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=-1) + self.smooth

        loss = 1 - (num / den)
        loss = (weight * loss).sum(dim=-1)
        # batch reduction
        if self.reduction == 'mean':
            return torch.mean(loss, dim=-1)
        elif self.reduction == 'sum':
            return torch.sum(loss, dim=-1)
        else:  # 'none'
            return loss


class GHM_Loss(BaseLoss):
    def __init__(self, bins=10, alpha=0.5, label_smoothing=0, **kwargs):
        '''
        bins: split to n bins
        alpha: hyper-parameter
        基于梯度的GHM损失平衡
        '''
        super(GHM_Loss, self).__init__(**kwargs)
        self._bins = bins
        self._alpha = alpha
        self.label_smoothing = label_smoothing
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.tensor(torch.floor(g * (self._bins - 0.0001))).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def compute(self, x, target):
        if x.shape != target.shape:
            target = F.one_hot(target, x.size(-1))
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros(self._bins, device=g.device)
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()
        bz, nc = x.size(0), x.size(1)
        N = (bz * nc)

        # 逐 batch 指数加权移动平均(EMA) 近似总样本下的梯度密度
        if self._last_bin_count is not None:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count

        self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd
        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    '''
        GHM_Loss for classification
    '''

    def __init__(self, bins, alpha, **kwargs):
        super(GHMC_Loss, self).__init__(bins, alpha, **kwargs)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class MultiGHMC_Loss(GHM_Loss):
    '''
        GHM_Loss for multi-class classification
    '''

    def __init__(self, bins, alpha, label_smoothing=0, **kwargs):
        super(MultiGHMC_Loss, self).__init__(bins, alpha, label_smoothing, **kwargs)

    def _custom_loss(self, x, target, weight):
        pt = F.softmax(x, dim=-1) + 1e-7  # avoid nan
        if self.label_smoothing > 0:
            c = x.size(-1)
            target = target * (1 - self.label_smoothing) + self.label_smoothing / c
        cross_entropy = -target * torch.log(pt) * weight  # cross entropy
        return cross_entropy.sum(dim=-1).mean(-1)

    def _custom_loss_grad(self, x, target):
        return torch.softmax(x, dim=-1).detach() - target


class GHMR_Loss(GHM_Loss):
    '''
        GHM_Loss for regression
    '''

    def __init__(self, bins, alpha, mu, **kwargs):
        super(GHMR_Loss, self).__init__(bins, alpha, **kwargs)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)


class LabelSmoothingCrossEntropy(BaseLoss):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100, **kwargs):
        super(LabelSmoothingCrossEntropy, self).__init__(**kwargs)
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def compute(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                                 ignore_index=self.ignore_index)


if __name__ == '__main__':
    loss = MultiGHMC_Loss(20, 0.5)
    x = torch.randn(2, 3)
    target = torch.tensor([[1, 0, 0], [0, 1, 0]]).float()
    print(loss(x, target))
