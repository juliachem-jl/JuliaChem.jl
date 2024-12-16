using JuliaChem.Shared.Constants.SCF_Keywords
mutable struct SCFOptions 
    density_fitting :: Bool 
    contraction_mode :: String
    load :: String 
    guess :: String
    energy_convergence :: Float64
    density_convergence :: Float64
    df_energy_convergence :: Float64
    df_density_convergence :: Float64
    max_iterations :: Int64
    df_max_iterations :: Int64
    df_exchange_n_blocks :: Int64
    df_screening_sigma :: Float64
    df_screen_exchange :: Bool
    #GPU algorithm specific
    df_force_dense :: Bool
    df_use_adaptive :: Bool
    num_devices :: Int64
    df_use_K_sym :: Bool
    df_K_sym_type :: String
end 

function create_default_scf_options()
    return SCFOptions(
        false, # density_fitting
        SCF_Keywords.ContractionMode.default,
        SCF_Keywords.Load.default,
        SCF_Keywords.Guess.default,
        SCF_Keywords.Convergence.energy_default,
        SCF_Keywords.Convergence.density_default,
        SCF_Keywords.Convergence.energy_default,
        SCF_Keywords.Convergence.density_default,
        SCF_Keywords.Convergence.max_iterations_default,
        SCF_Keywords.Convergence.df_max_iterations_default,
        SCF_Keywords.Screening.df_exchange_n_blocks_default,
        SCF_Keywords.Screening.df_sigma_default,
        SCF_Keywords.Screening.df_exchange_screen_mode_default,
        SCF_Keywords.GPUAlgorithms.default_df_force_dense,
        SCF_Keywords.GPUAlgorithms.df_use_adaptive_default,
        SCF_Keywords.GPUAlgorithms.default_num_devices,
        SCF_Keywords.GPUAlgorithms.df_use_K_sym_default,
        SCF_Keywords.GPUAlgorithms.df_K_sym_type,
        SCF_Keywords.GPUAlgorithms.df_adaptive_basis_limit
        )
end

function create_scf_options(scf_flags)


    guess::String = haskey(scf_flags, Guess.guess) ?
    scf_flags[Guess.guess] : Guess.default


    load::String = haskey(scf_flags, IntegralLoad.load) ? 
        scf_flags[IntegralLoad.load] : IntegralLoad.default

    contraction_mode::String = haskey(scf_flags, ContractionMode.contraction_mode) ?
        scf_flags[ContractionMode.contraction_mode] : ContractionMode.default

    do_density_fitting::Bool = haskey(scf_flags, SCFType.scf_type) ? 
        lowercase(scf_flags[SCFType.scf_type]) == SCFType.density_fitting : false

    energy_convergence = haskey(scf_flags, Convergence.energy) ?
        scf_flags[Convergence.energy] : Convergence.energy_default

    density_convergence = haskey(scf_flags, Convergence.density) ?
        scf_flags[Convergence.density] : Convergence.density_default


    if guess == Guess.density_fitting
        df_energy_convergence = haskey(scf_flags, Convergence.density_fitting_energy) ?
        scf_flags[Convergence.density_fitting_energy] : Convergence.energy_default

        df_density_convergence = haskey(scf_flags, Convergence.density_fitting_density) ?
        scf_flags[Convergence.density_fitting_density] : Convergence.density_default
    else
        df_energy_convergence = energy_convergence
        df_density_convergence = density_convergence
    end

  
  
    niter::Int = haskey(scf_flags, Convergence.max_iterations) ?
        scf_flags[Convergence.max_iterations] : Convergence.max_iterations_default

    df_exchange_n_blocks::Int = haskey(scf_flags, Screening.df_exchange_n_blocks) ? 
        scf_flags[Screening.df_exchange_n_blocks] : Screening.df_exchange_n_blocks_default 

    df_screening_sigma::Float64 = haskey(scf_flags, Screening.df_sigma) ? 
        scf_flags[Screening.df_sigma] : Screening.df_sigma_default

    df_screen_exchange::Bool = haskey(scf_flags, Screening.df_exchange_screen_mode) ? 
        scf_flags[Screening.df_exchange_screen_mode] : Screening.df_exchange_screen_mode_default

    df_niter = 0
    if do_density_fitting 
        df_niter = niter        
        
    elseif guess == Guess.density_fitting
        df_niter::Int = haskey(scf_flags, Convergence.df_max_iterations) ?
        scf_flags[Convergence.df_max_iterations] : Convergence.df_max_iterations_default        
    end

    df_force_dense = haskey(scf_flags, GPUAlgorithms.df_force_dense) ? 
        scf_flags[GPUAlgorithms.df_force_dense] : GPUAlgorithms.default_df_force_dense

    df_use_adaptive = haskey(scf_flags, GPUAlgorithms.df_use_adaptive) ? 
        scf_flags[GPUAlgorithms.df_use_adaptive] : GPUAlgorithms.df_use_adaptive_default
    
    df_num_devices = haskey(scf_flags, GPUAlgorithms.num_devices) ?
        scf_flags[GPUAlgorithms.num_devices] : GPUAlgorithms.default_num_devices
    
    df_use_K_sym = haskey(scf_flags, GPUAlgorithms.df_use_K_sym) ?
        scf_flags[GPUAlgorithms.df_use_K_sym] : GPUAlgorithms.df_use_K_sym_default

    df_K_sym_type = haskey(scf_flags, GPUAlgorithms.df_K_sym_type) ?
        scf_flags[GPUAlgorithms.df_K_sym_type] : GPUAlgorithms.df_K_sym_type_default
    
    df_adaptive_basis_limit = haskey(scf_flags, GPUAlgorithms.df_adaptive_basis_limit) ?
        scf_flags[GPUAlgorithms.df_adaptive_basis_limit] : GPUAlgorithms.df_adaptive_basis_limit_default

    return SCFOptions(
        do_density_fitting,
        contraction_mode,
        load,
        guess,
        energy_convergence,
        density_convergence,
        df_energy_convergence,
        df_density_convergence,
        niter,
        df_niter, 
        df_exchange_n_blocks,
        df_screening_sigma,
        df_screen_exchange,
        df_force_dense,
        df_use_adaptive,
        df_num_devices,
        df_use_K_sym,
        df_K_sym_type,
        df_adaptive_basis_limit
        )
end

function print_scf_options(options::SCFOptions)
    println("SCF Options:")
    println("------------------------------")
   
    
    println("Integral Load: ", options.load)
    
    println("Guess: ", options.guess)
    println("Energy Convergence: ", options.energy_convergence)
    println("Density Convergence: ", options.density_convergence)
    println("Max Iterations: ", options.max_iterations)
    println("--------------------------------")
    if options.density_fitting
        println("")
        println("Density Fitting Options:")
        println("--------------------------------")
        println("Contraction Mode: ", options.contraction_mode)
        println("DF Energy Convergence: ", options.df_energy_convergence)
        println("DF Density Convergence: ", options.df_density_convergence)
        if options.df_exchange_n_blocks == 0 && options.contraction_mode == "GPU"
            println("DF Exchange Block Width: adaptive")
        else
            println("DF Exchange Block Width: nbas ÷ ", options.df_exchange_n_blocks)
        end
        println("DF Screening Sigma: ", options.df_screening_sigma)
        println("DF Screen Exchange: ", options.df_screen_exchange)
        if options.contraction_mode == "GPU"
            println("DF Force Dense: ", options.df_force_dense)
            println("DF Use Adaptive: ", options.df_use_adaptive)
            println("DF Use K Symmetry: ", options.df_use_K_sym)
            println("DF K Symmetry Type: ", options.df_K_sym_type)
            println("DF Adaptive basis limit ", options.df_adaptive_basis_limit)
            println("DF number of GPUs: ", options.num_devices)
        end
        println("--------------------------------")
    end
end

export SCFOptions, create_default_scf_options, create_scf_options, print_scf_options