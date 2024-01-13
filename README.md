# Adaptive Topological Feature via Persistent Homology: Filtration Learning for Point Clouds (NeurIPS 2023)

Paper URL: https://arxiv.org/abs/2307.09259

We propose a framework that learns a filtration adaptively with the use of neural networks. 
In order to make the resulting persistent homology isometry-invariant, we develop a neural network architecture with such invariance. 

## Experiment for protein dataset

We utilize the protein dataset in Kovacev-Nikolic et al. (2016) consisting of dense configuration data of 14 types of proteins. 
The authors of the paper analyzed the maltose-binding protein (MBP), whose topological structure is important for investigating its biological role. 
They constructed a dynamic model of 370 essential points each for these 14 types of proteins and calculated the cross-correlation matrix $C$ for each type[^1]. 
They then define the distance matrix, called dynamical distance, by $D_{ij}=1-|C_{ij}|$, and they use them to classify the proteins into two classes, “open" and “close" based on their topological structure. 
In our experiments, we subsampled 60 points and used the distance matrices with the shape $60\times 60$ for each instance. 
We subsampled 1,000 times, half of which were from open and the rest were from closed. The off-diagonal elements of the distance matrix were perturbed by adding noise from a normal distribution with a standard deviation of 0.1.

The python code for this experiment is `scripts/dist_directly_slove.py`.
Please refer to `bash/execution_KNprotein.sh` for the usage. 

## Experiment for 3D CAD Dataset

ModelNet10 (Wu et al., 2015) is a dataset consisting of 3D-CAD data given by collections of polygons of 10 different kinds of objects, such as beds, chairs, and desks. 
We choose 1,000 point clouds from this dataset so that the resulting dataset includes 100 point clouds for each class. 
The corresponding point cloud was created with the sampling from each polygon with probability proportional to its area. 
Moreover, to evaluate the performance in the case where the number of points in each point cloud is small, we subsampled 128 points from each point cloud. 
The number of points is relatively small compared with the previous studies, but this setting would be natural from the viewpoint of practical applications. 
We added noise to the coordinates of each sampled point using a normal distribution with a standard deviation of 0.1.

The python code for this experiment is `scripts/coord_directly_solve.py`.
Please refer to `bash/execution_modelnet.sh` for the usage. 

[^1]: The correlation matrix C can be found in https://www.researchgate.net/publication/301543862_corr.