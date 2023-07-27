import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn

def consistency_cost(model,teacher_model,imgs,p_masks):
    output1=model(imgs)
    output2=teacher_model(imgs)
    loss=F.cross_entropy(output1,output2)
    return loss