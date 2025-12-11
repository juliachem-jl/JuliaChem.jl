# module load cudatoolkit
# module load craype-accel-nvidia80;
# export CRAY_ACCEL_TARGET=nvidia80;
# export MPICH_GPU_SUPPORT_ENABLED=1;
# export JULIA_CUDA_MEMORY_POOL=none;
# export JULIA_CUDA_USE_BINARYBUILDER=false;
module load julia/1.9.4
script_path="/global/homes/j/jhayes1/source/JuliaChem.jl/test/density-fitting-test.jl"

srun -N 1 -n 4 --gpus-per-task=1 julia --threads=64 $script_path &> ./testoutputs/water_4_rank_1gpu_1.log
