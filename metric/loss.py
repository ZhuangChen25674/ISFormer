import torch
import torch.nn as nn



class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes


    def forward(self, input, target):

        pred = torch.sigmoid(input)
        smooth = 1
        intersection = pred * target
        

        # Numerator Product
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -
                                                intersection.sum() + smooth)

        # Return average loss over classes and batch
        return 1 - loss.mean()


