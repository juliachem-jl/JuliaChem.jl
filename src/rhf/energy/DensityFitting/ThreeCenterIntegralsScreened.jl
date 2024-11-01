using Base.Threads
using LinearAlgebra
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared

three_center_integral_tag = 3000

function calculate_three_center_integrals_screened!(rank, n_ranks, 
    jeri_engine_thread, basis_sets, thead_integral_buffer, screened_pq_index_count, shell_screen_matrix, sparse_pq_index_map)
  
    nthreads = Threads.nthreads()
    basis_length = length(basis_sets.primary)
    
    rank_shell_indicies, 
    rank_basis_indicies, 
    rank_basis_index_map = static_load_rank_indicies_3_eri(rank, n_ranks, basis_sets) 

    rank_number_of_shells = length(rank_shell_indicies)
    n_indicies_per_thread = rank_number_of_shells÷nthreads

    rank_P = length(rank_basis_indicies) #the total number of indicies given to this rank
    three_center_integrals = zeros(Float64, (rank_P, screened_pq_index_count))

    Threads.@sync for thread in 1:nthreads
        Threads.@spawn begin     
            thread_index_offset = static_load_thread_index_offset(thread, n_indicies_per_thread)
            n_shells_to_process = static_load_thread_shell_to_process_count(thread, nthreads, rank_number_of_shells, n_indicies_per_thread)

            for shell_index in 1:n_shells_to_process
                aux_index = rank_shell_indicies[shell_index + thread_index_offset]
                for μ in 1:basis_length
                    for ν in 1:μ 
                        if shell_screen_matrix[μ, ν] == false
                            continue 
                        end
                        cartesian_index = CartesianIndex(aux_index, μ, ν)
                        engine = jeri_engine_thread[thread]
                        integral_buffer = thead_integral_buffer[thread]
                        calculate_three_center_integrals_kernel_screened!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer, sparse_pq_index_map, rank_basis_index_map)
                    end
                end    
            end
        end
    end
    return three_center_integrals
end


#calculate only the integrals that pass Schwarz screening
#sparse_pq_index_map is a map of the non screened pq pairs to where in the screned matrix they should go in the 1D pq index, could be dense or triangular 
@inline function calculate_three_center_integrals_kernel_screened!(three_center_integrals, engine, cartesian_index, basis_sets, integral_buffer, sparse_pq_index_map, rank_basis_index_map)
    s1, s2, s3,
    shell_1, shell_2, shell_3,
    shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, 
    bf_1_pos, bf_2_pos, bf_3_pos, 
    number_of_integrals = get_indexes_eri_block(cartesian_index, basis_sets)
    
    JERI.compute_eri_block_df(engine, integral_buffer, s1, s2, s3, number_of_integrals, 0)
    copy_integral_result_screened!(three_center_integrals, integral_buffer, bf_1_pos, bf_2_pos, bf_3_pos, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, sparse_pq_index_map, rank_basis_index_map)
    axial_normalization_factor_screened!(three_center_integrals, shell_1, shell_2, shell_3, shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, bf_1_pos, bf_2_pos, bf_3_pos, sparse_pq_index_map, rank_basis_index_map)

end


@inline function copy_integral_result_screened!(three_center_integrals, values, bf_1_pos, bf_2_pos, bf_3_pos, 
    shell_1_nbasis, shell_2_nbasis, shell_3_nbasis, 
    screen_index_matrix, rank_basis_index_map)
    values_index = 1

    for i in bf_1_pos:bf_1_pos+shell_1_nbasis-1 #auxiliary_basis index
        rank_i = rank_basis_index_map[i] #auxiliary_basis index for the rank 
        for j in bf_2_pos:bf_2_pos+shell_2_nbasis-1
            for k in bf_3_pos:bf_3_pos+shell_3_nbasis-1
                screened_index = screen_index_matrix[j,k]
                if screened_index == 0
                    values_index += 1
                    continue
                end
                three_center_integrals[rank_i,screened_index] = values[values_index]
                values_index += 1
            end
        end
    end
end
