filename=scripts/coord_directly_solve

dataset="ModelNetNoisy01"
num_points=128
pointnet=0
deepsets=0
pointmlp=1
matds=0
toporep=0
dtm=1
savedirname="result/${dataset}_n${num_points}_pn${pointnet}_ds${deepsets}_pm${pointmlp}_mds${matds}_tr${toporep}_dtm${dtm}"

mkdir $savedirname
python3 $filename.py $savedirname $dataset $num_points $pointnet $deepsets $pointmlp $matds $toporep $dtm