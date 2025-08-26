#!/bin/bash
#PBS -P cp23
#PBS -q gpuvolta
#PBS -l ncpus=96
#PBS -l ngpus=8
#PBS -l mem=100GB
#PBS -l walltime=48:00:00
#PBS -l wd
#PBS -l storage=gdata/cp23+scratch/cp23
NNODES=$((PBS_NCPUS / PBS_NCI_NCPUS_PER_NODE))
node_gpu=$((PBS_NGPUS / NNODES))
NGPUS=${PBS_NGPUS}
cur_host=$(cat $PBS_NODEFILE | head -n 1)
MASTER_NAME="${cur_host/\.gadi\.nci\.org\.au/}"
PORT=29501
for inode in $(seq 1 $PBS_NCI_NCPUS_PER_NODE $PBS_NCPUS); do
  if test $inode -eq 1 
  then
    JOB_RANK=0
  else
    JOB_RANK=1
  fi
  pbsdsh -n $inode -v ../tools/run_8gpus1.sh $NGPUS $MASTER_NAME $PORT $JOB_RANK $node_gpu &
done
wait