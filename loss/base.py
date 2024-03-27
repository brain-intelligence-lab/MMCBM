class BaseLoss:
    def __init__(self, num_classes=None, class_weight=None, loss_weight=1, reduction: str = 'mean', **kwargs):
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.reduction = reduction
        for key, value in kwargs.items():
            setattr(self, key, value)

    def compute(self, pre, inp):
        raise NotImplementedError

    def __call__(self, pre, inp):
        if self.loss_weight != 1:
            return self.loss_weight * self.compute(pre, inp)
        return self.compute(pre, inp)
