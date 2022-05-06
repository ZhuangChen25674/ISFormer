import torch
import numpy as np
import threading


class tp():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass):

        self.nclass = nclass
        self.lock = threading.Lock()
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.T = 0
        self.P = 0
        self.reset()

    def update(self, preds, labels):

        def evaluate_worker(self, label, pred):
            self.TP,self.TN,self.FP,self.FN,self.T,self.P = batch_pix_accuracy(
                pred, label)
            

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                       )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):

        PD = self.TP / (self.T)
        FA = self.FP / (self.P)

        return PD, FA

    def reset(self):

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.T = 0
        self.P = 0



def batch_pix_accuracy(output, target):
    # pre label  pytorch tensor

    if len(target.shape) == 3:
        target = torch.unsqueeze(target, axis=1).numpy().astype('int64') # T
    elif len(target.shape) == 4:
        target = target.cpu().numpy().astype('int64') # T
    else:
        raise ValueError("Unknown target dimension")
    # print("output.shape: ", output.shape)
    # print("target.shape: ", target.shape)
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output.detach().cpu().numpy() > 0).astype('int64') # P
    T = np.sum(target > 0) # T
    P = np.sum(target == 0)
    TP = np.sum((predict == target)*(target > 0)) # TP
    TN = np.sum((predict == target)*(target == 0))
    FP = np.sum((predict != target)*(target == 0)*(predict==1))
    FN = np.sum((predict != target)*(target == 1)*(predict==0))

    
    return TP,TN,FP,FN,T,P


def batch_intersection_union(output, target):
    #ped label pytorch tensor

    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass
    predict = (output.detach().cpu().numpy() > 0).astype('int64') # P
    if len(target.shape) == 3:
        target = torch.unsqueeze(target, axis=1).numpy().astype('int64') # T
    elif len(target.shape) == 4:
        target = target.cpu().numpy().astype('int64') # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target) # TP
    
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union



