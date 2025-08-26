#!/bin/bash -l
#PBS -P cp23
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=180GB
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -l storage=gdata/cp23+scratch/cp23
cd /scratch/cp23/sz1566/PROB
source /home/135/sz1566/miniconda3/bin/activate owdetr
sleep $((24 * 3600))
