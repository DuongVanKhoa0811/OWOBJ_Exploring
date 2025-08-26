#!/bin/bash
#PBS -P cp23
#PBS -q gpuvolta
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=180GB
#PBS -l walltime=48:00:00
#PBS -l wd
#PBS -l storage=gdata/cp23+scratch/cp23
GPUS_PER_NODE=4 ../tools/run_dist_launch.sh 4 configs/M_OWOD_BENCHMARK1.sh