#!/bin/bash
#
#SBATCH --job-name=s10_new_algo
#SBATCH --output=s10_new_algo-3-20.log
#SBATCH --error=s10_new_algo-3-20.err
#
#SBATCH --partition=haswell
#
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#
export JULIA_NUM_THREADS=20
export OMP_NUM_THREADS=20
#
/export/home/jackson/.julia/bin/mpiexecjl -np 3 julia --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes density-fitting-test.jl
