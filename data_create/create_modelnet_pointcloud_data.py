import torch
import numpy as np
from tqdm import tqdm

from lib.modelnet_process import *

sampling_time = 1
sampling_point_num = 2000
cls_num = 10
data_per_cls = 100
noise_level = 0.1

filename_data = f"data/ModelNetNoisy_C=10,N=100,T=1,K=2000,S={str(noise_level).replace('.', '')}_data"
filename_label = f"data/ModelNetNoisy_C=10,N=100,T=1,K=2000,S={str(noise_level).replace('.', '')}_label"

cls_list = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
train_cls_list = cls_list[:cls_num]

pointcloud_list = []
for cls in tqdm(train_cls_list):
    for i in range(1, 1+data_per_cls):
        with open(f"data/ModelNet10/{cls}/train/{cls}_{str(i).zfill(4)}.off", 'r') as f:
            verts, faces = read_off(f)
        for j in range(sampling_time):
            pointcloud = PointSampler(sampling_point_num)((verts, faces))

            avg = np.mean(pointcloud, axis=0)
            std = np.std(pointcloud, axis=0)
            pointcloud = (pointcloud - avg.reshape(1, 3)) / std.reshape(1, 3)

            pointcloud_list.append(torch.tensor(pointcloud).to(torch.float32))

data = torch.stack(pointcloud_list, axis=0)
data = data + torch.normal(0., noise_level, size=data.shape)
label = torch.tensor([i//(data.shape[0]//cls_num) for i in range(data.shape[0])]).to(torch.long)

data = data[:, random.sample(range(data.shape[1]), k=data.shape[1]), :]

torch.save(data, filename_data)
torch.save(label, filename_label)
print(data.shape, label.shape)