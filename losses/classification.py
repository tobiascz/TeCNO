import torch
import torch.nn.functional as F


def classification_loss(self, labels, y_classes):
    stages = y_classes.shape[0]
    clc_loss = 0
    weights = torch.Tensor(self.weights_train).type_as(y_classes)
    for j in range(
            stages):  ### make the interuption free stronge the more layers.
        p_classes = y_classes[j].squeeze().transpose(1, 0)
        ce_loss = F.cross_entropy(p_classes, labels.squeeze(), weight=weights)
        clc_loss += ce_loss
    clc_loss = clc_loss / (stages * 1.0)
    return clc_loss
