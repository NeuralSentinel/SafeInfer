import os, re, json
import torch, numpy as np

import sys
torch.set_grad_enabled(False)

from torch.nn import functional as F
import torch


class fvLayer(torch.nn.Module):
    def __init__(self, fv, device="cuda"):
        super(fvLayer, self).__init__()
        self.fv = fv

    def forward(self, x):
        idx = -1
        input_dtype = x.dtype
        # print("----------")
        # print("Input: ", x, x.shape)
        # print("Indx: ", x[0][idx, :].shape)
        # print("Tell here: ",isinstance(x, tuple))
        #if isinstance(x, tuple):
        #print("Pre sum: ",x)
        #print("FV vector: ", self.fv)
        x[0][idx, :] += 1.5 * self.fv.flatten().to("cuda:0")
        #print("After sum: ",x)
        return x
    
class model_with_FunctionVector(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, FV, EDIT_LAYER):
        for i in range(len(self.model.model.layers)):
            if i in EDIT_LAYER:
                self.model.model.layers[i].mlp = torch.nn.Sequential(self.model.model.layers[i].mlp, fvLayer(FV))
                #print(self.model.state_dict)
        return self.model