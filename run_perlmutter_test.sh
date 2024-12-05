#!/bin/bash
#SBATCH -A m4265
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH -c 64
#SBATCH -J S22_3
#SBATCH -o /global/homes/j/jhayes1/source/JuliaChem.jl/testoutputs/test_intermitent_dense_code.out

export JULIA_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64

module load cudatoolkit
module load julia/1.9.4

script="/global/homes/j/jhayes1/source/JuliaChem.jl/test/density-fitting-test.jl"

srun julia --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes $script