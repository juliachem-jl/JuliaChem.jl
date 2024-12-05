module load julia 

srun -N 2 -n 4 julia --project=/pscratch/sd/j/jhayes1/source/JuliaChem.jl/perl_cpu_env TinkerB_MPI.jl &> ./tinkerMPI.log
