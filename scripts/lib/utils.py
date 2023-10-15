import torch
import numpy as np
import seaborn as sns
import pandas as pd

def PI_value(grid, points, h=1, h_list=None, PI_weight="linear", density="gaussian"):
    if type(grid).__module__ != "torch":
        grid = torch.tensor(grid).to(torch.float32)

    ret = torch.zeros_like(grid[:, :, 0])
    
    if density == "gaussian":
        for p, q in points:
            dist = torch.stack(
                [
                    torch.stack(
                        [torch.linalg.norm(grid[i, j, :] - torch.stack([p, q], dim=0)) for j in range(grid.shape[1])]
                    ) for i in range(grid.shape[0])
                ], dim=0)
            if PI_weight == "linear":
                ret += (1 / (h**2)) * q * torch.exp(- (dist**2 / (2*(h**2))))
            elif PI_weight == "quad": 
                ret += (1 / (h**2)) * (q ** 2)* torch.exp(- (dist**2 / (2*(h**2))))
            elif PI_weight == "tanh":
                ret += (1 / (h**2)) * torch.tanh(q) * torch.exp(- (dist**2 / (2*(h**2))))
    elif density == "cone":
        for p, q in points:
            dist = torch.tensor([[torch.linalg.norm(grid[i, j, :] - torch.tensor([p, q])) for j in range(grid.shape[1])] for i in range(grid.shape[0])])
            if PI_weight == "linear":
                ret += (6 / (np.pi * (h**2))) * q * torch.fmin(h - dist, torch.zeros_like(dist))
            elif PI_weight == "quad": 
                ret += (6 / (np.pi * (h**2))) * (q ** 2)* torch.fmin(h - dist, torch.zeros_like(dist))
            elif PI_weight == "tanh":
                ret += (6 / (np.pi * (h**2))) * torch.tanh(q) * torch.fmin(h - dist, torch.zeros_like(dist))
    else:
        raise NotImplementedError
    return ret

def draw_heatmap(heatmap_array, ax, vmin=0.0, vmax=2.0):
    if type(heatmap_array).__module__ == "torch":
        heatmap_array = heatmap_array.detach().numpy()
        
    sns.heatmap(
        pd.DataFrame(heatmap_array.T).iloc[::-1], 
        cmap="jet", 
        xticklabels=True, 
        yticklabels=True, 
        vmin = vmin, 
        vmax = vmax, 
        ax = ax
    )

def random_rotation(X):
    theta = 2 * np.pi * np.random.rand(X.shape[0])
    if X.shape[2] == 3:
        rot_mat_list = [
            torch.tensor([
                [np.cos(theta[k]), -np.sin(theta[k]), 0.], 
                [np.sin(theta[k]),  np.cos(theta[k]), 0.], 
                [0., 0., 1.]
            ]).to(torch.float32) for k in range(X.shape[0])
        ]
    else:
        rot_mat_list = [
            torch.tensor([
                [np.cos(theta[k]), -np.sin(theta[k])], 
                [np.sin(theta[k]),  np.cos(theta[k])], 
            ]).to(torch.float32) for k in range(X.shape[0])
        ]
    rotated_X = torch.stack([torch.einsum("ij,kj->ki", A, X[l, :, :]) for l, A in enumerate(rot_mat_list)], dim=0)
    return rotated_X

def pointcloud_normalize(X):
    pc_list = []
    for i in range(X.shape[0]):
        avg = torch.mean(X[i, :, :], dim=0)
        std = torch.std(X[i, :, :], dim=0)
        pointcloud = (X[i, :, :] - avg.reshape(1, X.shape[2])) / std.reshape(1, X.shape[2])
        pc_list.append(pointcloud)
    data = torch.stack(pc_list, axis=0)
    
    return data
