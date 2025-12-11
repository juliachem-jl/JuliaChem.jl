#!/bin/bash
#
#SBATCH --job-name=s22_benchmark-1-36
#SBATCH --output=s22_benchmark-1-36.log
#SBATCH --error=s22_benchmark-1-36.err
#
#SBATCH --partition=haswell
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36

export JULIA_NUM_THREADS=36
export OMP_NUM_THREADS=36

#use sysimg
# julia -J"../SYSIMG/JuliaChem.so" --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes ./density-fitting-vs-rhf.jl

# use without sysimg
julia --check-bounds=no --math-mode=fast --optimize=3 --inline=yes --compiled-modules=yes ./density-fitting-vs-rhf.jl
