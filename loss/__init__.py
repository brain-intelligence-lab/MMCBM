from .cls_loss import *


class Loss(nn.Module):
    def __init__(self, loss_type='CrossEntropy', **kwargs):
        super(Loss, self).__init__()
        self.loss_type = loss_type
        self.loss = self.get_loss(loss_type, **kwargs)

    def get_loss(self, loss_type, **kwargs):
        if loss_type == 'CrossEntropy':
            return CrossEntropy(**kwargs)
        elif loss_type == 'MSE':
            return MSE(**kwargs)
        elif loss_type == 'Dice':
            return DiceLoss(**kwargs)
        elif loss_type == 'CrossFocal':
            return CrossFocal(**kwargs)
        elif loss_type == 'GHMC':
            return GHMC_Loss(**kwargs)
        elif loss_type == 'MultiGHMC':
            return MultiGHMC_Loss(**kwargs)
        elif loss_type == 'GHMR':
            return GHMR_Loss(**kwargs)
        elif loss_type == 'LabelSmoothingCrossEntropy':
            return LabelSmoothingCrossEntropy(**kwargs)
        else:
            raise ValueError(f'Loss type {loss_type} not supported')

    def forward(self, pre, inp):
        loss = 0
        if isinstance(pre, tuple):
            pre, loss = pre
        loss += self.loss(pre, inp)
        return loss
