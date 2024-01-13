filename=scripts/dist_directly_solve

### Data settings ###
# Name of the dataset. The python code will access to "data/{dataset}_data" and "data/{dataset}_label".
dataset="KNproteinNoisy01_C=7,T=500,K=60"

### PH settings: You can use zero or one of the following three options. ###
matds=0 # If 1, DistMatrixNet will be used instead of PH. If 0, this will not be used. 
toporep=0 # If 1, proposed method will be used. If 0, this will not be used. 
dtm=3 # If >= 1, DTM filtration with k=dtm will be used. Especially, if dtm=1, Rips filtration will be used. If 0, this will not be used. 

### Name of the directory to save ###
savedirname="result/${dataset}_mds${matds}_tr${toporep}_dtm${dtm}"

mkdir $savedirname
python3 $filename.py $savedirname $dataset $matds $toporep $dtm