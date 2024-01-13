filename=scripts/coord_directly_solve

### Data settings ###
# Name of the dataset. The python code will access to "data/{dataset}_data" and "data/{dataset}_label".
dataset="ModelNetNoisy01"
# Number of the points sampled from each point cloud
num_points=128

### DNN settings: You can use exactly one of the following three options. ###
pointnet=0
deepsets=0
pointmlp=1

### PH settings: You can use zero or one of the following three options. ###
matds=0 # If 1, DistMatrixNet will be used instead of PH. If 0, this will not be used. 
toporep=0 # If 1, proposed method will be used. If 0, this will not be used. 
dtm=1 # If >= 1, DTM filtration with k=dtm will be used. Especially, if dtm=1, Rips filtration will be used. If 0, this will not be used. 

### Name of the directory to save ###
savedirname="result/${dataset}_n${num_points}_pn${pointnet}_ds${deepsets}_pm${pointmlp}_mds${matds}_tr${toporep}_dtm${dtm}"

mkdir $savedirname
python3 $filename.py $savedirname $dataset $num_points $pointnet $deepsets $pointmlp $matds $toporep $dtm