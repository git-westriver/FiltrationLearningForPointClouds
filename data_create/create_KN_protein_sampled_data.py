import torch
import numpy as np
import random
import glob

sampling_time = 500
sampling_point_num = 60
type_num = 7
noise_size = 0.2

if noise_size == 0:
    filename_data = f"data/KNprotein_C={type_num},T={sampling_time},K={sampling_point_num}_data"
    filename_label = f"data/KNprotein_C={type_num},T={sampling_time},K={sampling_point_num}_label"
else:
    filename_data = f"data/KNproteinNoisy{str(noise_size).replace('.', '')}_C={type_num},T={sampling_time},K={sampling_point_num}_data"
    filename_label = f"data/KNproteinNoisy{str(noise_size).replace('.', '')}_C={type_num},T={sampling_time},K={sampling_point_num}_label"

type_list = [[], []]
for x in glob.glob("data/KN_corr/0_mbp_open/*.txt"):
    type_list[0].append(x.split("/")[-1].split(".txt")[0])
for x in glob.glob("data/KN_corr/0_mbp_closed/*.txt"):
    type_list[1].append(x.split("/")[-1].split(".txt")[0])
type_list = [type_list[i][:type_num] for i in range(2)]

print("Creating data...")
dist_list = [[], []]
for i, cls in enumerate(["open", "closed"]):
    for ty in type_list[i]: 
        corr = np.loadtxt(f"data/KN_corr/0_mbp_{cls}/{ty}.txt")
        dist_list[i].append(1 - np.abs(corr))

sampled_dist_list = []
for i in range(2):
    for k in range(sampling_time):
        type_idx = random.choice(range(len(dist_list[i])))
        sample_idx_list = random.sample(range(340), k=sampling_point_num)
        noise_matrix = torch.randn(sampling_point_num, sampling_point_num) * noise_size if noise_size > 0 else torch.zeros(sampling_point_num, sampling_point_num)
        noize_matrix = (noise_matrix + noise_matrix.t()) * np.sqrt(2) / 2
        sampled_dist_list.append(
            torch.tensor(
                dist_list[i][type_idx][sample_idx_list, :][:, sample_idx_list]
            ).to(torch.float32)
            + noize_matrix - torch.diag(torch.diag(noize_matrix))
        )

data = torch.stack(sampled_dist_list, dim=0)
torch.save(data, filename_data)

label = torch.tensor([0]*(data.shape[0]//2) + [1]*(data.shape[0]//2)).to(torch.long)
torch.save(label, filename_label)