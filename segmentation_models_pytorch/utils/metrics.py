from . import base
from . import functional as F
from ..base.modules import Activation
import torch
import numpy as np


class BaseMetric(base.Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0]) #.squeeze(1)
        y_pr = self.activation(y_pr).squeeze(1) # without aux_params
        #print(y_pr.shape, y_pr.min(), y_pr.max())
        #print('prediction', y_pr.shape)
        #print('gt', y_gt.shape)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

class mIoU(BaseMetric):
    __name__='miou'

    def forward(self, y_pr, y_gt):
        y_pr = torch.argmax(y_pr, dim=1).view(-1)
        y_gt = y_gt.view(-1)
        unique_classes = torch.unique(torch.cat([y_gt, y_pr]))
        iou_list = [F.iou(y_pr==cls, y_gt==cls, eps=self.eps) for cls in unique_classes]
        valid_iou = [x for x in iou_list if torch.isfinite(x)]
        return torch.mean(torch.stack(valid_iou)) if valid_iou else torch.tensor(float('nan'))

    
class mIoU2(base.Metric):
    __name__ = 'miou_score'
    
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        
    def forward(self, y_pr, y_gt):
        #y_pr = F.softmax(pred, dim=1)
        y_pr = torch.argmax(y_pr, dim=1) #.squeeze(1)
        iou_list = list()
        present_iou_list = list()
        
        y_pr = y_pr.view(-1)
        y_gt = y_gt.view(-1)
        for sem_class in range(0, 4):
            pred_inds = (y_pr == sem_class)
            target_inds = (y_gt == sem_class)
            if target_inds.long().sum().item() == 0:
                iou_now = float('nan')
            else: 
                intersection_now = (pred_inds[target_inds]).long().sum().item()
                union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
                iou_now = float(intersection_now) / float(union_now)
                present_iou_list.append(iou_now)
            iou_list.append(iou_now)
            #print(round(iou_list, 2))
        return torch.as_tensor(np.mean(present_iou_list))




class Fscore(base.Metric):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0])
        y_pr = self.activation(y_pr).squeeze(1)

        
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0])
        y_pr = self.activation(y_pr) #without aux_params
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0])
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
