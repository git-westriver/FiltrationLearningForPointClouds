import torch
import torch.nn as nn

from lib.pointnet import *

class View(nn.Module):
    def __init__(self, *target_shape):
        super(View, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view((-1,) + self.target_shape) 

class SumPool(nn.Module):
    def __init__(self, num_channels, original_shape):
        super(SumPool, self).__init__()
        self.num_channels = num_channels
        self.original_shape = original_shape

    def forward(self, input_data):
        out = input_data.view((-1,) + self.original_shape[:-1] + (self.num_channels, ))
        out = torch.sum(out, dim=-2)
        return out

class DeepSets(nn.Module):
    def __init__(self, input_dim, num_points, num_labels):
        super(DeepSets, self).__init__()
        self.input_dim = input_dim
        self.num_points = num_points
        self.num_labels = num_labels
        original_shape = (num_points, input_dim)
        
        self.main = nn.Sequential(
            nn.Flatten(0, 1),
            NonLinear(self.input_dim, 64),
            NonLinear(64, 64),
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            SumPool(1024, original_shape),
            NonLinear(1024, 512),
            nn.Dropout(p = 0.3),
            NonLinear(512, 256),
            nn.Dropout(p = 0.3),
            NonLinear(256, self.num_labels),
            )

    def forward(self, input_data):
        return self.main(input_data)

class MatrixDeepSets(nn.Module):
    def __init__(self, input_dim, num_points, num_labels, doub=False, **kwargs):
        super(MatrixDeepSets, self).__init__()
        self.input_dim = input_dim
        self.num_points = num_points
        self.num_labels = num_labels
        self.doub = doub

        if input_dim != 1: raise NotImplementedError
        
        if doub:
            self.main = nn.Sequential(
                nn.Flatten(0, 2),
                NonLinear(self.input_dim, 64, **kwargs),
                NonLinear(64, 128, **kwargs),
                NonLinear(128, 256, **kwargs),
                SumPool(256, (num_points, num_points, input_dim)),
                nn.Flatten(0, 1),
                NonLinear(256, 256, **kwargs),
                NonLinear(256, 256, **kwargs),
                NonLinear(256, 256, **kwargs),
                SumPool(256, (num_points, input_dim)),
                NonLinear(256, 128, **kwargs),
                nn.Dropout(p = 0.3),
                NonLinear(128, 64, **kwargs),
                nn.Dropout(p = 0.3),
                NonLinear(64, self.num_labels, **kwargs),
                )
        else:
            self.main = nn.Sequential(
                nn.Flatten(0, 2),
                NonLinear(self.input_dim, 64, **kwargs),
                NonLinear(64, 128, **kwargs),
                NonLinear(128, 256, **kwargs),
                SumPool(256, (num_points, num_points, input_dim)),
                nn.Flatten(0, 1),
                NonLinear(256, 128, **kwargs),
                nn.Dropout(p = 0.3),
                NonLinear(128, 64, **kwargs),
                nn.Dropout(p = 0.3),
                NonLinear(64, self.num_labels, **kwargs),
                View(self.num_points, self.num_labels), 
                )
    
    def forward(self, input_data):
        return self.main(input_data)