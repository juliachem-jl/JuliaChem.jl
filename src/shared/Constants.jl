module Constants

module SCF_Keywords
    module SCFType
        const scf_type = "scf_type"
        const rhf = "rhf" # Restricted Hartree Fock
        const density_fitting = "df" # Density Fitted Restricted Hartree Fock
    end

    module Screening
        const df_exchange_n_blocks = "df_exchange_n_blocks"
        const df_exchange_n_blocks_default = 0
        const df_exchange_n_blocks_cpu_default = 10
        const df_exchange_n_blocks_gpu_default = 2
        const df_sigma = "df_sigma"
        const df_sigma_default = 1E-5
        const df_exchange_screen_mode = "df_exchange_screen"
        const df_exchange_screen_mode_default = false
    end

    module Guess
        const guess = "guess"
        const default = "hcore"
        const hcore = "hcore"
        const density_fitting = "df" # density fitting with hcore guess before using full rhf 
        const sad = "sad"
    end

    module Convergence
        const density_fitting_energy = "df_dele"
        const energy = "dele"
        const energy_default = 1E-3

        const density_fitting_density = "df_rmsd"
        const density = "rmsd"
        const density_default = 1E-3

        const max_iterations = "niter"
        const max_iterations_default = 10

        const df_max_iterations = "df_niter"
        const df_max_iterations_default = 10
    end

    module ContractionMode 
        const contraction_mode = "contraction_mode"
        const default = "default"
        const dense = "dense" # use BLAS library 
        const screened = "screened" # default 
    end 

    module IntegralLoad 
        const load = "load"
        const default = "static"
        const sequential = "sequential"
        const dynamic = "dynamic"
    end

    module GPUAlgorithms
        const gpu_algorithm = "gpu_algorithm"
        const default = "GPU"
        const denseGPU = "denseGPU"
        const df_force_dense = "df_force_dense"
        const default_df_force_dense = false
        const df_use_adaptive = "df_use_adaptive"
        const df_use_adaptive_default = true # false == always use screening algorithm
        const num_devices = "num_devices"
        const default_num_devices = 1
        const df_use_K_sym = "df_use_K_sym"
        const df_use_K_sym_default = false 
        const df_K_sym_type = "df_K_sym_type"
        const df_K_sym_type_default = "square" 
        const df_K_sym_type_square = "square" 
        const df_K_sym_rect = "rect"
        const df_adaptive_basis_limit = "df_adaptive_basis_limit"
        const df_adaptive_basis_limit_default = 800
    end
    
    export SCFType, ContractionMode, IntegralLoad, Guess, Convergence, Screening, GPUAlgorithms
end 



export SCF, SCF_Keywords

end