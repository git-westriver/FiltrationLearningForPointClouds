import sys
import random
import time
import datetime
import pickle

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gudhi.dtm_rips_complex import DTMRipsComplex

from lib.pc_representation import *
from lib.scheduler import *

if __name__ == "__main__":
    start_time = time.time()
    args = sys.argv[1:]
    if not args:
        args = ["result", "ModelNetNoisy01_C=10,N=100,T=1,K=2000", 128, 1, 0, 0, 0, 0, 1]

    print("SETTINGS: ", args)
    ### Save Directory ###
    savedirname = args[0]
    ### Dataset ###
    dataset = args[1]
    num_points = int(args[2])
    ### Network Architecture ###
    pointnet = int(args[3])
    deepsets = int(args[4])
    pointmlp = int(args[5])
    matds = int(args[6])
    toporep = int(args[7])
    dtm = int(args[8])
    ### Optimization Hyper Parameters ###
    epoch_num = [1000, 100]
    first_warmup_epoch_num = 40
    second_warmup_epoch_num = 40
    batch_size = 40
    first_lr = 1e-2 if pointmlp else 2e-2
    second_lr = 1e-2
    ### pretrained models ###
    if pointnet: first_model_name = "pointnet"
    elif deepsets: first_model_name = "deepsets"
    elif pointmlp: first_model_name = "pointmlp"
    else: first_model_name = None

    if first_model_name is not None: 
        # You can set pretrained DNN model path here if necessary.
        # dnn_pretrained_model_path = f"result/pretrain_{dataset}_phase1_{first_model_name}" 
        dnn_pretrained_model_path = None

    ### Optional Parameters ###
    CV_num = 5
    CV_color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    plt.rcParams["font.size"] = 30
    method_toporep = "dist"

    all_X = torch.load(f"data/{dataset}_data").to(torch.float32)
    all_X = pointcloud_normalize(all_X[:, :num_points, :])
    all_y = torch.load(f"data/{dataset}_label")
    data_num = all_X.shape[0]
    input_dim = all_X.shape[2]
    task_output_dim = 10
    criterion = nn.CrossEntropyLoss()
    task = "cls"

    ### Cross Validation ###
    valid_task_loss_list = [[[0]*epoch_num[0], [0]*epoch_num[1]] for _ in range(CV_num)]
    if task == "cls": valid_task_acc_list = [[[0]*epoch_num[0], [0]*epoch_num[1]] for _ in range(CV_num)]
    data_idx = list(range(data_num))
    data_idx.sort(key=lambda i: (all_y[i], i))

    ## 1st Phase ##
    if (first_model_name is not None) and (dnn_pretrained_model_path is None):
        print("=== 1st Phase ===")
        for CV_idx in range(CV_num):
            print(f"--- Cross Validation: {CV_idx + 1} / {CV_num} ---")
            train_idx = [data_idx[i] for i in range(data_num) if i % 5 != CV_idx]
            trainX = all_X[train_idx, :, :]
            trainy = all_y[train_idx]
            valid_idx = [data_idx[i] for i in range(data_num) if i % 5 == CV_idx]
            validX = all_X[valid_idx, :, :]
            validy = all_y[valid_idx]
            train_num = trainX.shape[0]
            valid_num = validX.shape[0]

            if task == "cls":
                print("Train Data Distribution: ", {j: int(sum([trainy[i] == j for i in range(train_num)])) for j in range(task_output_dim)})
                print("Valid Data Distribution: ", {j: int(sum([validy[i] == j for i in range(valid_num)])) for j in range(task_output_dim)})
        
            first_feature_dim = 16 * (pointnet + deepsets + pointmlp)
            first_feature_net = PCFeatureNet(input_dim, num_points, pointnet=pointnet, deepsets=deepsets, pointmlp=pointmlp)
            if task == "cls":
                first_task_solver = nn.Sequential(
                    nn.Linear(first_feature_dim, task_output_dim), 
                    nn.Softmax(dim=1),
                )
            else:
                first_task_solver = nn.Sequential(
                    nn.Linear(first_feature_dim, task_output_dim),
                )
                
            first_feature_net.eval()
            first_task_solver.eval()
            with torch.no_grad():
                valid_out = first_task_solver(first_feature_net(validX))
                valid_loss = float(criterion(valid_out, validy))
                if task == "cls":
                    valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
                    print(f"Initial valid_loss = {valid_loss:.3f}, Initial valid_acc = {valid_acc:.1f}")
                else:
                    print(f"Initial valid_loss = {valid_loss:.3f}")
            
            first_opt = torch.optim.Adam(list(first_feature_net.parameters()) + list(first_task_solver.parameters()), lr=first_lr)
            first_scheduler = TransformerLR(first_opt, warmup_epochs=first_warmup_epoch_num) if first_warmup_epoch_num is not None else None
            train_toporep_points_list = [[] for _ in range(trainX.shape[0])] if toporep else None
            for epoch in range(epoch_num[0]):
                first_feature_net.train()
                first_task_solver.train()
                    
                epoch_loss_list = []
                idx_list = random.sample(range(train_num), train_num)
                for idx in range(train_num//batch_size):
                    first_opt.zero_grad()
                    batch_idx_list = idx_list[batch_size * idx: batch_size * (idx+1)]
                    batchX = random_rotation(trainX[batch_idx_list, :, :])
                    batch_out = first_task_solver(first_feature_net(batchX))
                    batch_loss = criterion(batch_out, trainy[batch_idx_list])
                    batch_loss.backward()
                    first_opt.step()
                    epoch_loss_list.append(float(batch_loss))
                
                first_feature_net.eval()
                first_task_solver.eval()
                with torch.no_grad():
                    valid_out = first_task_solver(first_feature_net(validX))
                    valid_loss = float(criterion(valid_out, validy))
                    if task == "cls":
                        valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
                        print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}, valid_acc = {valid_acc:.1f}", flush=True)
                    else:
                        print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}", flush=True)
                    
                    valid_task_loss_list[CV_idx][0][epoch] = valid_loss
                    if task == "cls": valid_task_acc_list[CV_idx][0][epoch] = valid_acc
            
                if first_scheduler is not None: 
                    first_scheduler.step()
            
            torch.save(first_feature_net.state_dict(), f"{savedirname}/first_feature_net_CVidx={CV_idx}.pth")
            torch.save(first_task_solver.state_dict(), f"{savedirname}/first_task_solver_CVidx={CV_idx}.pth")
            dnn_pretrained_model_path = savedirname

    if dtm:
        dtm_points_list = []
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
            
    print("=== 2nd Phase ===")
    for CV_idx in range(CV_num):
        print(f"--- Cross Validation: {CV_idx + 1} / {CV_num} ---")
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
        
        second_feature_dim = 16 * (pointnet + deepsets + pointmlp + matds + (dtm >= 1) + toporep)
        if first_model_name is not None:
            first_feature_net = PCFeatureNet(input_dim, num_points, pointnet=pointnet, deepsets=deepsets, pointmlp=pointmlp)
            first_feature_net.load_state_dict(torch.load(f"{dnn_pretrained_model_path}/first_feature_net_CVidx={CV_idx}.pth"))
            for param in first_feature_net.parameters(): param.requires_grad = False
            first_feature_net.eval()
        second_feature_net = PCFeatureNet(input_dim, num_points, matds=matds, ph=(dtm or toporep))
        if toporep:
            toporep_net = TopoRep(num_points, 16, reducer=False, perslay=False, method=method_toporep)
        else: toporep_net = None

        if task == "cls":
            second_task_solver = nn.Sequential(
                nn.Linear(second_feature_dim, task_output_dim), 
                nn.Softmax(dim=1),
            )
        else:
            second_task_solver = nn.Sequential(
                nn.Linear(second_feature_dim, task_output_dim),
            )

        second_feature_net.eval()
        second_task_solver.eval()
        if toporep: toporep_net.eval()
        with torch.no_grad():
            if dtm:
                valid_points_list = valid_dtm_points_list
            elif toporep:
                valid_points_list = toporep_net.get_pd_points_list(validX)
            else:
                valid_points_list = None
            valid_out = second_task_solver(second_feature_net(
                validX, pd_points_list=valid_points_list, out_vec=first_feature_net(validX) if first_model_name is not None else None
            ))
            valid_loss = float(criterion(valid_out, validy))
            if task == "cls":
                valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
                print(f"Initial valid_loss = {valid_loss:.3f}, Initial valid_acc = {valid_acc:.1f}")
            else:
                print(f"Initial valid_loss = {valid_loss:.3f}")
        
        second_opt = torch.optim.Adam(list(second_feature_net.parameters()) + list(second_task_solver.parameters()) + (list(toporep_net.parameters()) if toporep else []), lr=second_lr)
        second_scheduler = TransformerLR(second_opt, warmup_epochs=second_warmup_epoch_num) if second_warmup_epoch_num is not None else None
        for epoch in range(epoch_num[1]):
            second_feature_net.train()
            second_task_solver.train()
            if toporep: toporep_net.train()

            epoch_loss_list = []
            idx_list = random.sample(range(train_num), train_num)
            for idx in range(train_num//batch_size):
                second_opt.zero_grad()
                batch_idx_list = idx_list[batch_size * idx: batch_size * (idx+1)]
                batchX = random_rotation(trainX[batch_idx_list, :, :])
                if dtm:
                    batch_points_list = [train_dtm_points_list[i] for i in batch_idx_list]
                elif toporep:
                    batch_points_list = toporep_net.get_pd_points_list(batchX)
                else:
                    batch_points_list = None
                batch_out = second_task_solver(second_feature_net(
                    batchX, pd_points_list=batch_points_list, out_vec=first_feature_net(batchX) if first_model_name is not None else None
                ))
                batch_loss = criterion(batch_out, trainy[batch_idx_list])
                batch_loss.backward()
                second_opt.step()
                epoch_loss_list.append(float(batch_loss))
            
            second_feature_net.eval()
            second_task_solver.eval()
            if toporep: toporep_net.eval()
            with torch.no_grad():
                if dtm:
                    valid_points_list = valid_dtm_points_list
                elif toporep:
                    valid_points_list = toporep_net.get_pd_points_list(validX)
                else:
                    valid_points_list = None
                valid_out = second_task_solver(second_feature_net(
                    validX, pd_points_list=valid_points_list, out_vec=first_feature_net(validX) if first_model_name is not None else None
                ))
                valid_loss = float(criterion(valid_out, validy))
                if task == "cls":
                    valid_acc = 100 * float(sum(torch.max(valid_out, dim=1).indices == validy) / valid_num)
                    print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}, valid_acc = {valid_acc:.1f}", flush=True)
                else:
                    print(f"Epoch {epoch}: average_loss = {sum(epoch_loss_list)/len(epoch_loss_list):.3f}, valid_loss = {valid_loss:.3f}", flush=True)
                
                valid_task_loss_list[CV_idx][1][epoch] = valid_loss
                if task == "cls": valid_task_acc_list[CV_idx][1][epoch] = valid_acc
        
            if second_scheduler is not None: 
                second_scheduler.step()

        torch.save(second_feature_net.state_dict(), f"{savedirname}/second_feature_net_CVidx={CV_idx}.pth")
        torch.save(second_task_solver.state_dict(), f"{savedirname}/second_task_solver_CVidx={CV_idx}.pth")
        if toporep: torch.save(toporep_net.state_dict(), f"{savedirname}/toporep_CVidx={CV_idx}.pth")

    print(f"=== Summary ===")
    print(f"Time consuming: {datetime.timedelta(seconds=time.time() - start_time)}")
    for phase in range(2):
        print(f"--- Phase {1 + phase} ---")
        avg = {}
        std = {}
        if task == "cls":
            avg["FinalAccuracy"] = np.mean([valid_task_acc_list[i][phase][-1] for i in range(CV_num)])
            std["FinalAccuracy"] = np.std([valid_task_acc_list[i][phase][-1] for i in range(CV_num)], ddof=0)
            print("Distribution of Final Accuracy", [f"{float(valid_task_acc_list[i][phase][-1]):.1f}" for i in range(CV_num)])
        
        avg["Loss"] = np.mean([valid_task_loss_list[i][phase][-1] for i in range(CV_num)])
        std["Loss"] = np.std([valid_task_loss_list[i][phase][-1] for i in range(CV_num)], ddof=0)
        print("Distribution of Valid Loss", [f"{float(valid_task_loss_list[i][phase][-1]):.3f}" for i in range(CV_num)])

        for k in avg.keys():
            _avg = avg[k]
            _std = std[k]
            print(f"Average {k}: {_avg}")
            print(f"Std of {k}: {_std}")
            print(f"{_avg:.3f} ± {_std:.3f}")
            print(f"{_avg:.2f} ± {_std:.2f}")
            print(f"{_avg:.1f} ± {_std:.1f}")
    
    fig = plt.figure(figsize=(40, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ymin, ymax = 1000, 0
    for i in range(CV_num):
        ax1.plot(list(range(epoch_num[0])), valid_task_loss_list[i][0], color=CV_color_list[i])
        ax2.plot(list(range(epoch_num[0], epoch_num[0] + epoch_num[1])), valid_task_loss_list[i][1], color=CV_color_list[i])
        ymin, ymax = min(ymin, ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ymax, ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    fig.subplots_adjust(left=0.05, right=0.975, bottom=0.05, top=0.95, wspace=0.15, hspace=0.15)
    fig.savefig(f"{savedirname}/loss_history.png")
    if task == "cls":
        fig = plt.figure(figsize=(40, 12))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ymin, ymax = 1000, 0
        for i in range(CV_num):
            ax1.plot(list(range(epoch_num[0])), valid_task_acc_list[i][0], color=CV_color_list[i])
            ax2.plot(list(range(epoch_num[0], epoch_num[0] + epoch_num[1])), valid_task_acc_list[i][1], color=CV_color_list[i])
            ymin, ymax = min(ymin, ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ymax, ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)
        fig.subplots_adjust(left=0.05, right=0.975, bottom=0.05, top=0.95, wspace=0.15, hspace=0.15)
        fig.savefig(f"{savedirname}/accuracy_history.png")

    with open(f"{savedirname}/loss_history", "wb") as f:
        pickle.dump(valid_task_loss_list, f)
    if task == "cls":
        with open(f"{savedirname}/accuracy_history", "wb") as f:
            pickle.dump(valid_task_acc_list, f)