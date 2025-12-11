using JuliaChem.JCModules

using JuliaChem.Shared
using JuliaChem.Shared.JCTC

function set_user_scf_options_data!(timing::JCTiming,scf_options::SCFOptions)
    set_options(timing.user_options, scf_options)
end

function set_scf_options_data!(timing::JCTiming,scf_options::SCFOptions)
    set_options(timing.options, scf_options)
end

function set_options(dictionary::Dict{String, String}, scf_options::SCFOptions)
    dictionary[JCTC.density_fitting] = string(scf_options.density_fitting)
    dictionary[JCTC.contraction_mode] = scf_options.contraction_mode
    dictionary[JCTC.load] = scf_options.load
    dictionary[JCTC.guess] = scf_options.guess
    dictionary[JCTC.energy_convergence] = string(scf_options.energy_convergence)
    dictionary[JCTC.density_convergence] = string(scf_options.density_convergence)
    dictionary[JCTC.df_energy_convergence] = string(scf_options.df_energy_convergence)
    dictionary[JCTC.df_density_convergence] = string(scf_options.df_density_convergence)
    dictionary[JCTC.max_iterations] = string(scf_options.max_iterations)
    dictionary[JCTC.df_max_iterations] = string(scf_options.df_max_iterations)
    dictionary[JCTC.df_exchange_n_blocks] = string(scf_options.df_exchange_n_blocks)
    dictionary[JCTC.df_screening_sigma] = string(scf_options.df_screening_sigma)
    dictionary[JCTC.df_screen_exchange] = string(scf_options.df_screen_exchange)
    dictionary[JCTC.df_force_dense] = string(scf_options.df_force_dense)
    dictionary[JCTC.df_use_adaptive] = string(scf_options.df_use_adaptive)
    dictionary[JCTC.num_devices] = string(scf_options.num_devices)
    dictionary[JCTC.df_use_K_sym] = string(scf_options.df_use_K_sym)
    dictionary[JCTC.df_K_sym_type] = string(scf_options.df_K_sym_type)
end

function set_threads_and_ranks!(timing::JCTiming, n_threads::Int, n_ranks::Int)
    timing.non_timing_data[JCTC.n_threads] = string(n_threads)
    timing.non_timing_data[JCTC.n_ranks] = string(n_ranks)
end

function set_converged!(timing::JCTiming,scf_converged::Bool, iterations::Int, energy::Float64)
    timing[JCTC.converged] = string(scf_converged)
    timing[JCTC.n_iterations] = string(iterations)
    timing.scf_energy = energy
end

function set_basis_info!(jc_timing::JCTiming, basis::Basis, aux_basis::Union{Basis, Nothing})
    jc_timing.non_timing_data[JCTC.n_basis_functions] = string(basis.norb)
    if !isnothing(aux_basis)
        jc_timing.non_timing_data[JCTC.n_auxiliary_basis_functions] = string(aux_basis.norb)
    else
        jc_timing.non_timing_data[JCTC.n_auxiliary_basis_functions] = "0"
    end
    jc_timing.non_timing_data[JCTC.n_electrons] = string(basis.nels)
    jc_timing.non_timing_data[JCTC.n_occupied_orbitals] = string(basis.nels÷2)
end

function set_converge_properties!(jc_timing, scf_converged, n_iterations, scf_energy)
    jc_timing.converged = scf_converged
    jc_timing.non_timing_data[JCTC.n_iterations] = string(n_iterations)
    jc_timing.scf_energy = scf_energy
end

export set_scf_options_data!, set_user_scf_options_data!, set_threads_and_ranks!, set_converged!, set_basis_info!, set_converge_properties!