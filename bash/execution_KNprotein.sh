#!/bin/bash 
#PBS -k o 
#PBS -l nodes=dijon03:ppn=1,walltime=10000:00
#PBS -N SampleTorqueJob
filename=scripts/dist_directly_solve

dataset="KNproteinNoisy01"
matds=0
toporep=0
dtm=3
savedirname="result/${dataset}_mds${matds}_tr${toporep}_dtm${dtm}"

mkdir $savedirname
python3 $filename.py $savedirname $dataset $matds $toporep $dtm