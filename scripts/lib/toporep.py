import numpy as np
from gudhi.weighted_rips_complex import WeightedRipsComplex
import torch
import torch.nn as nn

from lib.utils import *
from lib.pointnet import *
from lib.deepsets import *

class TopoRep(nn.Module):
    def __init__(self, num_points, num_labels, **kwargs):
        super(TopoRep, self).__init__()
        
        ### necessary parameters ###
        self.num_points = num_points
        self.num_labels = num_labels

        ### optional parameters ###
        self.method = kwargs.get("method", "dist")
        self.input_format = kwargs.get("input_format", "coord")
        self.reducer = kwargs.get("reducer", False)
        self.perslay = kwargs.get("perslay", not self.reducer)
        
        ### Networks ###
        if self.method == "dist":
            self.weight_net = Dist2Weight(self.num_points, **kwargs)
        elif self.method == "dist_transformer":
            self.weight_net = Dist2WeightTransformer(self.num_points, **kwargs)
        else:
            raise NotImplementedError
        if self.reducer:
            self.reducer = nn.Sequential( 
                nn.Linear(self.PI_grid_num ** 2, self.num_labels), 
                nn.BatchNorm1d(self.num_labels), 
            )
        else:
            self.reducer = nn.Identity()
        if self.perslay:
            self.perslay = PersLay(num_labels=self.num_labels, **kwargs)
        else:
            self.perslay = None
    
    def _get_pd_points(self, dist: torch.Tensor, weight: torch.Tensor):
        filt = WeightedRipsComplex(dist.detach().numpy(), weight)
        simplex_tree = filt.create_simplex_tree(max_dimension=2)
        simplex_tree.compute_persistence()
        pers_pair = simplex_tree.persistence_pairs()

        _dist = dist.clone()
        _dist += torch.stack([weight for _ in range(_dist.shape[0])], dim=0)
        _dist += torch.stack([weight for _ in range(_dist.shape[0])], dim=1)
        
        points = []
        for x in pers_pair:
            if len(x[0]) == 2:
                birth = _dist[x[0][0], x[0][1]]
                death = _dist[x[1][0], x[1][1]]
                death = torch.max(death, _dist[x[1][2], x[1][0]])
                death = torch.max(death, _dist[x[1][1], x[1][2]])
                points.append((birth, death - birth))
            elif len(x[0]) >= 3:
                raise NotImplementedError
        
        return points
    
    def get_pd_points_list(self, X: torch.Tensor):
        """
        X: torch.Tensor (num_data x num_points x point_dim)
        """
        if self.input_format == "dist":
            dist = X.clone()
        else:
            dist = torch.cdist(X, X)

        weight = self.weight_net(dist, dist)[:, :, 0]
        points_list = [self._get_pd_points(dist[i, :, :], weight[i, :]) for i in range(dist.shape[0])]

        return points_list

    
    def _get_heatmap_tensors(self, dist, weight):
        grid = torch.tensor(
            [
                [
                    [x, y] for y in list(np.linspace(0, self.PI_max, self.PI_grid_num))
                ] for x in list(np.linspace(0, self.PI_max, self.PI_grid_num))
            ]
        ).to(torch.float32)
        heatmap_tensor_list = []
        for i in range(dist.shape[0]):
            points = self._get_pd_points(dist[i, :, :], weight[i, :])
            heatmap = PI_value(grid, points, h=self.PI_h, PI_weight=self.PI_weight)
            heatmap_tensor_list.append(heatmap)

        heatmap_tensors = torch.stack(heatmap_tensor_list, dim=0)
        return heatmap_tensors

    def forward(self, X):
        if self.input_format == "dist":
            dist = X.clone()
        else:
            dist = torch.cdist(X, X)

        weight = self.weight_net(dist, dist)[:, :, 0]
        if self.perslay is None:
            heatmap_tensors = self._get_heatmap_tensors(dist, weight)
            heatmap_tensors_flatten = heatmap_tensors.view(-1, self.PI_grid_num ** 2)
            representation = self.reducer(heatmap_tensors_flatten)
        else:
            points_list = [self._get_pd_points(dist[i, :, :], weight[i, :]) for i in range(dist.shape[0])]
            representation = self.perslay(points_list)

        return representation
        
class Dist2Weight(nn.Module):
    def __init__(self, num_points, **kwargs):
        super(Dist2Weight, self).__init__()
        self.num_points = num_points

        self.get_pc_feature = MatrixDeepSets(1, num_points, 16, doub=True, **kwargs)
        self.get_point_feature = MatrixDeepSets(1, num_points, 8, doub=False, **kwargs)

        ### optional parameters ###
        self.last_layer_bn = kwargs.get("last_layer_bn", False)

        self.mlp_weight = nn.Sequential(
            NonLinear(24, 256), 
            NonLinear(256, 512), 
            NonLinear(512, 256), 
            NonLinear(256, 1, batch_norm=self.last_layer_bn), 
        )
    
    def forward(self, dist_mat, dist_vec):
        """
        Input
        dist_mat: batch_size x pc_point_num x pc_point_num
        dist_vec: batch_size x N x pc_point_num (N: number of points to calculate the weight)
        """
        N = dist_vec.shape[1]

        dist_mat = dist_mat.unsqueeze(dim=3)
        pc_feature = self.get_pc_feature(dist_mat) # batch_size x feature_dim

        dist_vec = dist_vec.unsqueeze(dim=3)
        point_feature = self.get_point_feature(dist_vec) # batch_size x N x feature_dim

        pc_feature_dup = torch.stack([pc_feature]*N, dim=1) # batch_size x N x feature_dim
        concat_feature = torch.cat([pc_feature_dup, point_feature], dim=2)

        out = concat_feature.view(-1, concat_feature.shape[-1])
        out = self.mlp_weight(out)
        out = out.view(-1, self.num_points, out.shape[-1])

        return out

class Dist2WeightTransformer(nn.Module):
    def __init__(self, num_points, **kwargs):
        super(Dist2WeightTransformer, self).__init__()
        self.num_points = num_points
        ### optional parameters ###
        self.last_layer_bn = kwargs.get("last_layer_bn", False)

        self.get_point_feature = MatrixDeepSets(1, num_points, 32, doub=False, **kwargs)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)
        self.fc = nn.Sequential(nn.Linear(32, 1), nn.ReLU(inplace=True))
    
    def forward(self, dist_mat, dist_vec=None):
        """
        Input
        dist_mat: batch_size x pc_point_num x pc_point_num
        dist_vec: *not used*
        """
        dist_mat = dist_mat.unsqueeze(dim=3)
        point_feature_seq = self.get_point_feature(dist_mat) # batch_size x N x feature_dim
        out = self.fc(self.transformer(point_feature_seq))
        return out

class PersLay(nn.Module):
    def __init__(self, num_labels, **kwargs):
        super(PersLay, self).__init__()
        ### optional parameters ###
        self.num_obsrv = kwargs.get("num_obsrv", 10)

        self.theta = nn.Parameter(torch.rand(self.num_obsrv, 2) * 4)
        self.fc = nn.Linear(self.num_obsrv, num_labels)
    
    def forward(self, points_list):
        h = 0.5
        out_list = []
        for L in points_list:
            if L:
                points_tensor = torch.stack([torch.stack(points, dim=0) for points in L], dim=0)
                exp_dist_to_points = torch.exp(- (torch.cdist(self.theta, points_tensor) ** 2) / (2*(h**2)))
                out_list.append(torch.sum(exp_dist_to_points, dim=1))
            else:
                out_list.append(torch.zeros(self.num_obsrv))
        out = torch.stack(out_list, dim=0)
        out = self.fc(out)   
        return out