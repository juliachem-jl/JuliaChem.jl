mutable struct ScreeningData
    sparse_pq_index_map
    basis_function_screen_matrix
    sparse_p_start_indices
    non_screened_p_indices_count
    screened_triangular_indicies
    shell_screen_matrix
    screened_triangular_indicies_to_2d
    block_screen_matrix::Array{Bool,2}
    blocks_to_calculate::Array{Int,1}
    exchange_batch_indexes::Array{Tuple{Int,Int}}
    non_zero_ranges::Array{Array{UnitRange{Int}}}
    non_zero_sparse_ranges::Array{Array{UnitRange{Int}}}
    triangular_indices_count::Int
    screened_indices_count::Int
    K_block_width::Int
end

mutable struct SCFData
    D
    D_tilde
    two_electron_fock
    coulomb_intermediate
    density
    occupied_orbital_coefficients
    non_zero_coefficients # GTFOCK paper equation 4 "In order to use optimized matrix multiplication library functions to compute W(i, Q), for each p, we need a dense matrix consisting of the nonzero rows of C(r, i)."
    density_array
    J
    K
    k_blocks
    k_non_square_blocks #buffers for the non-square blocks of K
    bottom_corner_k_block #buffer for the last non-square blocks of K
    screening_data::ScreeningData
    gpu_data::SCFGPUData
    μ::Int
    occ::Int
    A::Int
    scf_iteration::Int
    lower_triangle_length::Int
end


function SCFData(gpu_data::SCFGPUData)
    sd = ScreeningData([], [], [], [], [], [], [], falses(1, 1), zeros(Int, 0), Array{Tuple{Int,Int}}(undef, 0),
        Array{Array{UnitRange{Int}}}(undef, 0), Array{Array{UnitRange{Int}}}(undef, 0), 0, 0, 0)
    return SCFData([], [], [], [], [],[], [], [],[], [], [], [], [], sd, gpu_data, 0, 0, 0, 0, 0)
end

export SCFData, ScreeningData