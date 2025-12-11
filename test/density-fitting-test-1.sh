#!/bin/bash
#
#SBATCH --job-name=test_par_issues_1
#SBATCH --output=test_par_issues_1.log
#SBATCH --error=test_par_issues_1.err
#
#SBATCH --partition=haswell
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
export JULIA_NUM_THREADS=8
export OMP_NUM_THREADS=8
#
/export/home/jackson/.julia/bin/mpiexecjl -np 1 julia --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes density-fitting-test.jl
