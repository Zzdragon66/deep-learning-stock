import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    """The quantile loss"""

    def __init__(self, quantiles):
        for quantile in quantiles:
            assert quantile < 1 and quantile > 0
        super().__init__()
        self.quantiles = quantiles

    def forward(self, prediciton, target):
        # assume the prediction contains N, 2
        # target contains N, 1
        error = target - prediciton
        loss = None 
        for i, quantile in enumerate(self.quantiles):
            if loss is None:
                loss = torch.max((quantile - 1) * error[:, i], quantile * error[:, i])
            else:
                loss += torch.max((quantile - 1) * error[:, i], quantile * error[:, i]) 
        return torch.mean(loss)
        