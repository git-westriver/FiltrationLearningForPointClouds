import sys
import random
import time
import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gudhi.dtm_rips_complex import DTMRipsComplex

from lib.pc_representation import *
from lib.pointnet import *
from lib.scheduler import *

if __name__ == "__main__":
    start_time = time.time()
    args = sys.argv[1:]
    if not args:
        args = ["result/sample", "KNprotein", 0, 1, 0]

    ### Save Directory ###
    savedirname = args[0]
    ### Dataset ###
    dataset = args[1]
    ### Network Architecture ###
    matds = int(args[2])
    toporep = int(args[3])
    dtm = int(args[4])
    ### Optimization Hyper Parameters ###
    epoch_num = 200
    warmup_epoch_num = 10
    batch_size = 40
    lr = 2e-2
    ### Regularization ###
    lamb = 0
    reg_ord = 1
    ### Optional Parameters ###
    CV_num = 5
    CV_color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    plt.rcParams["font.size"] = 30
    method_toporep = "dist"

    input_format = "dist"
    all_X = torch.load(f"data/{dataset}_C=7,T=500,K=60_data").to(torch.float32)
    all_y = torch.load(f"data/{dataset}_C=7,T=500,K=60_label")
    data_num = all_X.shape[0]
    num_points = all_X.shape[1]
    output_dim = 2
    criterion = nn.CrossEntropyLoss()
    task = "cls"

    if dtm:
        dtm_points_list = []

        if input_format == "dist":
            dist = all_X.clone()
        else:
            dist = torch.cdist(all_X, all_X)

        for i in range(data_num):
            filt = DTMRipsComplex(distance_matrix=dist[i, :, :].detach().numpy(), k=dtm)
            simplex_tree = filt.create_simplex_tree(max_dimension=2)
            barcode = simplex_tree.persistence()

            points = []
            for x in barcode:
                if x[0] == 1:
                    points.append((torch.tensor(x[1][0]).to(torch.float32), torch.tensor(x[1][1] - x[1][0]).to(torch.float32)))
            dtm_points_list.append(points)  

    ### 5-fold Cross Validation ###
    valid_task_loss_list = [[0]*epoch_num for _ in range(CV_num)]
    if task == "cls": valid_task_acc_list = [[0]*epoch_num for _ in range(CV_num)]
    data_idx = list(range(data_num))
    data_idx.sort(key=lambda i: (all_y[i], i))
    
    for CV_idx in range(CV_num):
        print(f"=== Cross Validation: {CV_idx + 1} / 5 ===")
        train_idx = [data_idx[i] for i in range(data_num) if i % 5 != CV_idx]
        trainX = all_X[train_idx, :, :]
        trainy = all_y[train_idx]
        valid_idx = [data_idx[i] for i in range(data_num) if i % 5 == CV_idx]
        validX = all_X[valid_idx, :, :]
        validy = all_y[valid_idx]
        if dtm:
            train_dtm_points_list = [dtm_points_list[i] for i in train_idx]
            valid_dtm_points_list = [dtm_points_list[i] for i in valid_idx]
        train_num = trainX.shape[0]
        valid_num = validX.shape[0]

        if task == "cls":
            print("Train Data Distribution: ", {j: int(sum([trainy[i] == j for i in range(train_num)])) for j in range(output_dim)})
            print("Test Data Distribution: ", {j: int(sum([validy[i] == j for i in range(valid_num)])) for j in range(output_dim)})

        feature_dim = 16
        if toporep: 
            toporep_net = TopoRep(num_points, 16, 
                                  reducer=False, perslay=False, method=method_toporep, 
                                  input_format="dist")
            for param in toporep_net.weight_net.parameters(): torch.nn.init.normal_(param, mean=0.0, std=1e-4)
        feature_net = PCFeatureNet(None, num_points, matds_dist=matds, ph=(dtm or toporep))
        task_solver = nn.Sequential(
            nn.Linear(feature_dim, output_dim), 
            nn.Softmax(dim=1),
        )
        
        feature_net.eval()
        task_solver.eval()
        if toporep: toporep_net.eval()
        with torch.no_grad():
            if dtm:
                valid_points_list = valid_dtm_points_list
            elif toporep:
                valid_points_list = toporep_net.get_pd_points_list(validX)
            else:
                valid_points_list = None
            valid_out = task_solver(feature_net(validX, pd_points_list=valid_points_list))
            valid_loss = float(criterion(valid_out, validy))
            if task == "cls":
                valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
                print(f"Initial valid_loss = {valid_loss}, Initial valid_acc = {valid_acc}")
            else:
                print(f"Initial valid_loss = {valid_loss}")
        
        opt = torch.optim.Adam(list(feature_net.parameters()) + list(task_solver.parameters()), lr=lr)
        scheduler = TransformerLR(opt, warmup_epochs=warmup_epoch_num) if warmup_epoch_num is not None else None
        for epoch in range(epoch_num):
            feature_net.train()
            task_solver.train()
            if toporep: toporep_net.train()

            epoch_loss_list = []
            idx_list = random.sample(range(train_num), train_num)
            for idx in range(train_num//batch_size):
                opt.zero_grad()
                batch_idx_list = idx_list[batch_size * idx: batch_size * (idx+1)]
                batchX = trainX[batch_idx_list, :, :]
                if dtm:
                    batch_points_list = [train_dtm_points_list[i] for i in batch_idx_list]
                elif toporep:
                    batch_points_list = toporep_net.get_pd_points_list(batchX)
                else:
                    batch_points_list = None
                batch_out = task_solver(feature_net(batchX, pd_points_list=batch_points_list))
                batch_loss = criterion(batch_out, trainy[batch_idx_list])
                if lamb > 0:
                    if feature_net.perslay is None: 
                        raise Exception("PersLay is not used. ")
                    batch_loss += lamb * torch.linalg.norm(feature_net.perslay.fc.weight, ord=reg_ord)
                batch_loss.backward()
                opt.step()
                epoch_loss_list.append(float(batch_loss))
            
            feature_net.eval()
            task_solver.eval()
            if toporep: toporep_net.eval()
            with torch.no_grad():
                if dtm:
                    valid_points_list = valid_dtm_points_list
                elif toporep:
                    valid_points_list = toporep_net.get_pd_points_list(validX)
                else:
                    valid_points_list = None
                valid_out = task_solver(feature_net(validX, pd_points_list=valid_points_list))
                valid_loss = float(criterion(valid_out, validy))
                if task == "cls":
                    valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
                    print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}, valid_acc = {valid_acc:.1f}", 
                          flush=True)
                else:
                    print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}", flush=True)

                valid_task_loss_list[CV_idx][epoch] = valid_loss
                if task == "cls": valid_task_acc_list[CV_idx][epoch] = valid_acc

            if scheduler is not None:
                scheduler.step()

        torch.save(feature_net.state_dict(), f"{savedirname}/feature_net_CVidx={CV_idx}.pth")
        torch.save(task_solver.state_dict(), f"{savedirname}/task_solver_CVidx={CV_idx}.pth")
        if toporep: torch.save(toporep_net.state_dict(), f"{savedirname}/toporep_CVidx={CV_idx}.pth")

    print(f"=== Summary ===")
    print(f"Time consuming: {datetime.timedelta(seconds=time.time() - start_time)}")
    avg = {}
    std = {}
    if task == "cls":
        avg["FinalAccuracy"] = np.mean([valid_task_acc_list[i][-1] for i in range(CV_num)])
        std["FinalAccuracy"] = np.std([valid_task_acc_list[i][-1] for i in range(CV_num)], ddof=0)
        print("Distribution of Final Accuracy", [f"{float(valid_task_acc_list[i][-1]):.1f}" for i in range(CV_num)])
    
    avg["FinalLoss"] = np.mean([valid_task_loss_list[i][-1] for i in range(CV_num)])
    std["FinalLoss"] = np.std([valid_task_loss_list[i][-1] for i in range(CV_num)], ddof=0)
    print("Distribution of Valid Loss", [f"{float(valid_task_loss_list[i][-1]):.3f}" for i in range(CV_num)])

    for k in avg.keys():
        _avg = avg[k]
        _std = std[k]
        print(f"Average {k}: {_avg}")
        print(f"Std of {k}: {_std}")
        print(f"{_avg:.3f} ± {_std:.3f}")
        print(f"{_avg:.2f} ± {_std:.2f}")
        print(f"{_avg:.1f} ± {_std:.1f}")
    
    fig = plt.figure(figsize=(40, 12))
    ax = fig.add_subplot(1, 1, 1)
    ymin, ymax = 1000, 0
    ax.plot(list(range(epoch_num)), valid_task_loss_list[i], color=CV_color_list)
    fig.savefig(f"{savedirname}/loss_history.png")
    if task == "cls":
        fig = plt.figure(figsize=(40, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(list(range(epoch_num)), valid_task_acc_list[i], color=CV_color_list[i])
        fig.savefig(f"{savedirname}/accuracy_history.png")
    
    with open(f"{savedirname}/loss_history", "wb") as f:
        pickle.dump(valid_task_loss_list, f)
    if task == "cls":
        with open(f"{savedirname}/accuracy_history", "wb") as f:
            pickle.dump(valid_task_acc_list, f)
