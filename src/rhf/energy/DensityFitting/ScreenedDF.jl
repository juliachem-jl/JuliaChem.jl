using Base.Threads
using LinearAlgebra
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared
using Serialization
using ThreadPinning 
using Serialization
using JuliaChem.Shared.JCTC
using JuliaChem.Shared.Constants

@inline function twoD_to1Dindex(i, j, p)
    return (i - 1) * p + j
end


function get_screening_metadata!(scf_data, sigma, jeri_engine_thread, two_center_integrals, basis_sets, jc_timing)
    screening_time = @elapsed begin
        max_P_P = get_max_P_P(two_center_integrals)
        scf_data.screening_data.shell_screen_matrix,
        scf_data.screening_data.basis_function_screen_matrix,
        scf_data.screening_data.sparse_pq_index_map = schwarz_screen_itegrals_df(scf_data, sigma, max_P_P, basis_sets, jeri_engine_thread)
    end 

    screning_metadata_time = @elapsed begin 

        basis_function_screen_matrix = scf_data.screening_data.basis_function_screen_matrix
        scf_data.screening_data.non_screened_p_indices_count = zeros(Int64, scf_data.μ)
        scf_data.non_zero_coefficients = Vector{Array}(undef, scf_data.μ)
        scf_data.screening_data.screened_indices_count = sum(basis_function_screen_matrix)
        scf_data.screening_data.sparse_p_start_indices = zeros(Int64, scf_data.μ)
        scf_data.screening_data.non_zero_ranges = Vector{Array{UnitRange{Int}}}(undef, scf_data.μ)
        scf_data.screening_data.non_zero_sparse_ranges = Vector{Array{UnitRange{Int}}}(undef, scf_data.μ)

        Threads.@threads for pp in 1:scf_data.μ # can this be done in parallel?? 
            first_index = 1
            while scf_data.screening_data.sparse_p_start_indices[pp] == 0
                if scf_data.screening_data.basis_function_screen_matrix[first_index, pp] != 0
                    scf_data.screening_data.sparse_p_start_indices[pp] = scf_data.screening_data.sparse_pq_index_map[first_index, pp]
                    break
                end
                first_index += 1
            end

            scf_data.screening_data.non_screened_p_indices_count[pp] = sum(view(basis_function_screen_matrix, :, pp))
            scf_data.non_zero_coefficients[pp] = zeros(scf_data.occ, scf_data.screening_data.non_screened_p_indices_count[pp])


            scf_data.screening_data.non_zero_ranges[pp] = Array{UnitRange{Int}}(undef, 0)
            scf_data.screening_data.non_zero_sparse_ranges[pp] = Array{UnitRange{Int}}(undef, 0)

            start_index = 0
            end_index = 0
            non_zero_index = 1
            for r in 1:scf_data.μ
                if scf_data.screening_data.basis_function_screen_matrix[r, pp]
                    if start_index == 0 
                        start_index = r 
                    end
                    end_index = r
                end
                if start_index != 0 && (!scf_data.screening_data.basis_function_screen_matrix[r, pp] || r == scf_data.μ) 
                    push!(scf_data.screening_data.non_zero_ranges[pp], start_index:end_index)
                    range_length = end_index - start_index + 1
                    push!(scf_data.screening_data.non_zero_sparse_ranges[pp], non_zero_index:(non_zero_index + range_length - 1))
                    non_zero_index += range_length 
                    
                    start_index = 0
                    end_index = 0
                end
            end
        end
    end
    jc_timing.timings[JCTC.screening_time] = screening_time
    jc_timing.timings[JCTC.screening_metadata_time] = screning_metadata_time
    jc_timing.timings[JCTC.screened_indices_count] = scf_data.screening_data.screened_indices_count

end


function df_rhf_fock_build_screened!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions, jc_timing::JCTiming) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}

    occupied_orbital_coefficients = permutedims(occupied_orbital_coefficients, (2, 1))

    if iteration == 1
        n_ranks = MPI.Comm_size(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        two_eri_time = @elapsed two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        s_metadata_time = @elapsed get_screening_metadata!(scf_data, scf_options.df_screening_sigma, jeri_engine_thread, two_center_integrals, basis_sets, jc_timing)
        
        j_ab_inv_time = @elapsed begin 
            LAPACK.potrf!('L', two_center_integrals)
            LAPACK.trtri!('L', 'N', two_center_integrals)
            J_AB_invt = two_center_integrals
        end
        B_time = 0.0
        three_eri_time = 0.0
        if n_ranks > 1 #todo update this to reduce communication?
            three_eri_time = @elapsed three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options, scf_data, rank, n_ranks, true, false)
            B_time = @elapsed calculate_B_multi_rank(scf_data, J_AB_invt, three_center_integrals, basis_sets, jeri_engine_thread_df, scf_options, jc_timing)
        else
            three_eri_time = @elapsed scf_data.D = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options,
                scf_data, rank, n_ranks, true, false)
            B_time = @elapsed BLAS.trmm!('L', 'L', 'N', 'N', 1.0, J_AB_invt, scf_data.D)    
        end
        # deallocate unneeded memory
        two_center_integrals = zeros(0)
        J_AB_invt = zeros(0)
        rank_Q_indicies = size(scf_data.D,1) 
        scf_data.D_tilde = zeros(Float64, (rank_Q_indicies, scf_data.occ, scf_data.μ))
        scf_data.coulomb_intermediate = zeros(Float64, rank_Q_indicies)

        scf_data.J = zeros(Float64, scf_data.screening_data.screened_indices_count)
        scf_data.K = zeros(Float64, size(scf_data.two_electron_fock))
        scf_data.density = zeros(Float64, (scf_data.μ, scf_data.μ))
        scf_data.density_array = zeros(Float64, scf_data.screening_data.screened_indices_count)


        #save the basis function screen matrix to the shared timing object for debugging
        calculate_exchange_block_screen_matrix(scf_data, scf_options, 
            SCF_Keywords.Screening.df_exchange_n_blocks_cpu_default,
            jc_timing)

        jc_timing.timings[JCTC.two_eri_time] = two_eri_time
        jc_timing.timings[JCTC.screening_metadata_time] = s_metadata_time
        jc_timing.timings[JCTC.form_J_AB_inv_time] = j_ab_inv_time
        jc_timing.timings[JCTC.B_time] = B_time
        jc_timing.timings[JCTC.three_eri_time] = three_eri_time
        jc_timing.non_timing_data[JCTC.contraction_algorithm] = "screened cpu"
    end
    calculate_exchange_screened!(scf_data, scf_options, occupied_orbital_coefficients, jc_timing, iteration)
    calculate_coulomb_screened(scf_data, occupied_orbital_coefficients, jc_timing, iteration)
end

function calculate_B_multi_rank(scf_data, J_AB_INV, three_center_integrals, basis_sets, jeri_engine_thread_df, scf_options, jc_timing::JCTiming)
    comm = MPI.COMM_WORLD
    this_rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    
    pq = size(three_center_integrals, 2)

    #divide the B_Q indicies that will go to each rank 
    #(these are different than the three_eri_rank_indicies indicies which are based on the static or dynamic load balancing)
    load_balance_indicies = [static_load_rank_indicies_3_eri(rank_index, n_ranks, basis_sets) for rank_index in 0:n_ranks-1]
    three_eri_rank_indicies = load_balance_indicies[this_rank+1][2]
    this_rank_B_Q_index_range = load_balance_indicies[this_rank+1][2]
    
    this_rank_Q_length = length(this_rank_B_Q_index_range)

    max_rank_n_aux_indicies = 0
    for rank_index in 0:n_ranks-1
        max_rank_n_aux_indicies = max(max_rank_n_aux_indicies, length(load_balance_indicies[rank_index+1][2]))
    end

    scf_data.D = zeros(Float64, (this_rank_Q_length, pq))
    B_temp_buffer = zeros(Float64, (max_rank_n_aux_indicies*pq)) 
    alpha = 1.0
    beta = 0.0


    MPI_time = 0.0
    this_rank_gemm = 0.0
    other_rank_gemm = 0.0
    add_time = 0.0
    println("Calculating B: BLAS num threads: ", BLAS.get_num_threads())
    for recieve_rank in 0:n_ranks-1
        recieve_rank_B_Q_index_range = load_balance_indicies[recieve_rank+1][2]
        recieve_rank_J_AB_INV = J_AB_INV[recieve_rank_B_Q_index_range, three_eri_rank_indicies] #this allocates memory perhaps needs to be done another way
            
        if recieve_rank == this_rank        
            this_rank_gemm = @elapsed BLAS.gemm!('N', 'N', 1.0, recieve_rank_J_AB_INV, three_center_integrals, 0.0, scf_data.D)
            for send_rank in 0:n_ranks-1
                if send_rank == this_rank
                    continue
                end
                MPI_time += @elapsed reduce_B_this_rank(B_temp_buffer, length(scf_data.D), this_rank, send_rank)     
                temp_buffer = view(B_temp_buffer, 1:length(scf_data.D))
                reshaped_temp_buffer = reshape(temp_buffer, (this_rank_Q_length, pq))
                add_time += @elapsed scf_data.D .+= reshaped_temp_buffer
            end
        else
            recieve_rank_index_range_length = length(recieve_rank_B_Q_index_range)
            B_buffer_view = view(B_temp_buffer, 1:(recieve_rank_index_range_length*pq))

            pointerA = pointer(recieve_rank_J_AB_INV, 1)
            pointerB = pointer(three_center_integrals, 1)
            pointerC = pointer(B_temp_buffer, 1)
            M = recieve_rank_index_range_length
            N = pq
            K = size(recieve_rank_J_AB_INV, 2)

            other_rank_gemm += @elapsed call_gemm!(Val(false), Val(false), M, N, K, alpha, pointerA, pointerB, beta, pointerC)
            MPI_time += @elapsed reduce_B_other_rank(B_temp_buffer,recieve_rank_index_range_length*pq, this_rank, recieve_rank)                
        end
    end

    println("reduction time: ", MPI_time)
    println("this rank gemm time: ", this_rank_gemm)
    println("other rank gemm time: ", other_rank_gemm)
    println("add time: ", add_time)

end

function reduce_B_this_rank(B, length_of_B, rank, other_rank)
    println("rank: $rank, recieving from rank: $other_rank")
    comm = MPI.COMM_WORLD
    max_value = (2^31 - 1)-1
    top_index = max_value
    start_index = 1
    while true
        top_index = min(top_index, length_of_B)
        MPI.Recv!(view(B, start_index:top_index), comm; source=other_rank)
        if top_index == length_of_B
            break
        end
        start_index = top_index + 1
        top_index += max_value

    end    
end

function reduce_B_other_rank(B, length_of_B, rank, other_rank)
    println("rank: $rank, sending to rank: $other_rank")
    comm = MPI.COMM_WORLD
    max_value = (2^31 - 1) -1
    top_index = max_value
    start_index = 1
    while true
        top_index = min(top_index, length_of_B)
        MPI.Send(view(B, start_index:top_index), comm; dest=other_rank)
        if top_index == length_of_B
            break
        end
        start_index = top_index + 1
        top_index += max_value
    end    
end

function calculate_exchange_screened!(scf_data, scf_options, occupied_orbital_coefficients, jc_timing::JCTiming, iteration)
    W_time = @elapsed calculate_W_screened(scf_data, occupied_orbital_coefficients)
    
    K_time = @elapsed begin
        if scf_options.df_screen_exchange
            calculate_K_lower_diagonal_block(scf_data, scf_options)
        else
            calculate_K_lower_diagonal_block_no_screen(scf_data, scf_options)
        end
    end

    jc_timing.timings[JCTiming_key(JCTC.W_time, iteration)] = W_time    
    jc_timing.timings[JCTiming_key(JCTC.K_time, iteration)] = K_time
end

function calculate_W_screened(scf_data, occupied_orbital_coefficients)
    
    p = scf_data.μ
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    n_threads = Threads.nthreads()
    dynamic_p = n_threads + 1
    dynamic_lock = Threads.ReentrantLock()

    M = size(scf_data.D,1)
    N = scf_data.occ
    alpha = 1.0
    beta = 0.0

    linear_indicesB = LinearIndices(scf_data.D)
    linear_indicesW = LinearIndices(scf_data.D_tilde)

   
    Threads.@sync for thread in 1:n_threads
        Threads.@spawn begin
            pp = thread
            K = 1
            while pp <= p
                non_zero_r_index = 1
                for r in 1:p
                    if scf_data.screening_data.basis_function_screen_matrix[r, pp]
                        scf_data.non_zero_coefficients[pp][:, non_zero_r_index] .= view(occupied_orbital_coefficients, :, r)
                        non_zero_r_index += 1
                    end
                end
                K = scf_data.screening_data.non_screened_p_indices_count[pp]
                A_ptr = pointer(scf_data.D, linear_indicesB[1, scf_data.screening_data.sparse_p_start_indices[pp]])
                B_ptr = pointer(scf_data.non_zero_coefficients[pp], 1)
                C_ptr = pointer(scf_data.D_tilde, linear_indicesW[1, 1, pp])
                call_gemm!(Val(false), Val(true), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)

                lock(dynamic_lock) do
                    if dynamic_p <= p
                        pp = dynamic_p
                        dynamic_p += 1
                    else
                        pp = p + 1
                    end
                end
            end
        end
    end
    BLAS.set_num_threads(blas_threads)
end

function calculate_K_small(scf_data)

    M = scf_data.μ 
    N = scf_data.μ
    K = size(scf_data.D_tilde,1)*scf_data.occ

    A_ptr = pointer(scf_data.D_tilde, 1)
    B_ptr = pointer(scf_data.D_tilde, 1)
    C_ptr = pointer(scf_data.two_electron_fock, 1)

    call_gemm!(Val(true), Val(false), M, N, K, -1.0, A_ptr, B_ptr, 0.0, C_ptr) #it might not be necessary to do this with call gemm but it isn't going to hurt and keeps things consistent

end

function copy_screened_density_to_array(scf_data)
    Threads.@threads for index in CartesianIndices(scf_data.density)
        if !scf_data.screening_data.basis_function_screen_matrix[index[1], index[2]] || index[2] > index[1]
            continue
        end            
        if index[1] != index[2]
            scf_data.density_array[scf_data.screening_data.sparse_pq_index_map[index[1], index[2]]] = 2.0*scf_data.density[index] # symmetric multiplication 
        else
            scf_data.density_array[scf_data.screening_data.sparse_pq_index_map[index[1], index[2]]] = scf_data.density[index]
        end
    end
end

function calculate_coulomb_screened(scf_data, occupied_orbital_coefficients, jc_timing::JCTiming, iteration)
    density_time = @elapsed begin 
        BLAS.gemm!('T', 'N', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, scf_data.density)
        copy_screened_density_to_array(scf_data)
    end

    rank_Q = size(scf_data.D_tilde, 1)

    sparse_pq_index_map = scf_data.screening_data.sparse_pq_index_map

    blas_threads = BLAS.get_num_threads()
    println("BLAS threads: ", blas_threads)
    V_time = @elapsed begin 
        last_blas_add_time = 0.0
        p = scf_data.μ
        # scf_data.coulomb_intermediate .= 0.0
        BLAS.set_num_threads(1)

        n_threads = min(Threads.nthreads(), p-1)
        num_p_per_thread = p ÷ n_threads

        vv_time = @elapsed Threads.@threads for tt in 1:n_threads
            thread_time = @elapsed begin 
                p_thread_start = (tt - 1) * num_p_per_thread + 1
                p_thread_end =  tt * num_p_per_thread
                if tt == n_threads
                    p_thread_end = p-1
                end
                thread_V = view(
                    view(scf_data.D_tilde, :,:, p_thread_start), 1:rank_Q)
                thread_V .= 0.0
                for pp in p_thread_start:p_thread_end
                    range_start = scf_data.screening_data.sparse_p_start_indices[pp]
                    range_end = scf_data.screening_data.sparse_p_start_indices[pp+1] - 1
                    BLAS.gemv!('N', 1.0, 
                        view(scf_data.D, :, range_start:range_end), 
                        view(scf_data.density_array, range_start:range_end),
                        1.0, thread_V) 
                end
                if tt == 1
                    last_blas_add_time = @elapsed begin
                        BLAS.gemv!('N', 1.0, 
                        view(scf_data.D, :, size(scf_data.D, 2)),
                        view(scf_data.density_array, scf_data.screening_data.screened_indices_count:scf_data.screening_data.screened_indices_count),
                        0.0, scf_data.coulomb_intermediate)
                    end
                end
            end
        end

        v_add_time = @elapsed begin 
            for t in 1:n_threads
                p_thread_start = (t - 1) * num_p_per_thread + 1

                axpy!(1.0, 
                view(view(scf_data.D_tilde, :,:, p_thread_start), 1:rank_Q),
                scf_data.coulomb_intermediate)
            end 
        end

        println("vv time: ", vv_time)
        println("v add time: ", v_add_time)
        println("last blas add time: ", last_blas_add_time)

    end


    J_time = @elapsed begin
        # do symm J 
        scf_data.J .= 0.0
        Threads.@threads for pp in 1:(p-1) #todo use call_gemv to remove view usage?
            range_start = sparse_pq_index_map[pp, pp]
            range_end = scf_data.screening_data.sparse_p_start_indices[pp+1]-1
            BLAS.gemv!('T', 2.0,
                view(scf_data.D, :, range_start:range_end),
                scf_data.coulomb_intermediate,
                1.0, view(scf_data.J, range_start:range_end))
            if pp == 1
                range_start = scf_data.screening_data.screened_indices_count
                range_end = scf_data.screening_data.screened_indices_count
                BLAS.gemv!('T', 2.0,
                    view(scf_data.D, :, size(scf_data.D, 2)),
                    scf_data.coulomb_intermediate,
                    1.0, view(scf_data.J, range_start:range_end))
            end
        end
        copy_screened_coulomb_to_fock!(scf_data, scf_data.J, scf_data.two_electron_fock)
    end
   
    BLAS.set_num_threads(blas_threads)
    println("BLAS threads: ", blas_threads)

    jc_timing.timings[JCTiming_key(JCTC.density_time,iteration)] = density_time
    jc_timing.timings[JCTiming_key(JCTC.V_time,iteration)] = V_time
    jc_timing.timings[JCTiming_key(JCTC.J_time,iteration)] = J_time
end

function copy_screened_coulomb_to_fock!(scf_data, J, fock)

    Threads.@threads for index in CartesianIndices(scf_data.two_electron_fock)
        if !scf_data.screening_data.basis_function_screen_matrix[index] || index[2] > index[1]
            continue
        end
        fock[index] += J[scf_data.screening_data.sparse_pq_index_map[index]]
        if index[1] != index[2]
            fock[index[2], index[1]] = fock[index]
        end
    end
end


function get_triangle_matrix_length(n)::Int
    return n * (n + 1) ÷ 2
end

function calculate_exchange_block_screen_matrix(scf_data, scf_options, default_n_blocks, jc_timing::JCTiming)
    n_threads = Threads.nthreads()  

    if scf_options.df_exchange_n_blocks == 0
        scf_options.df_exchange_n_blocks = default_n_blocks
    end

    K_block_width = 0

    if scf_data.μ < 100 #if the # of basis functions is small just do a dense calculation with one block
        K_block_width = scf_data.μ
        scf_options.df_exchange_n_blocks = 1
        println("WARNING: K_block_width is less than 64, this may not be optimal for performance, K_block_with set to $K_block_width, df_exchange_n_blocks set to 1")

    else
        K_block_width = scf_data.μ ÷ scf_options.df_exchange_n_blocks

        # if K_block_width < 64
        #     for n_blocks in scf_options.df_exchange_n_blocks:-1:1
        #         bw = scf_data.μ ÷ n_blocks
        #         if bw >= 64
        #             K_block_width = bw
        #             scf_options.df_exchange_n_blocks = n_blocks
        #             break
        #         end
        #     end
     
        #     println("WARNING: K_block_width is less than 64, this may not be optimal for performance, K_block_with set to $K_block_width, df_exchange_n_blocks set to $(scf_options.df_exchange_n_blocks)")
        # end
    end

    lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_n_blocks)

    scf_data.screening_data.K_block_width = K_block_width
    
    jc_timing.non_timing_data[JCTC.total_exchange_blocks] = string(lower_triangle_length)
    jc_timing.non_timing_data[JCTC.df_exchange_n_blocks] = string(scf_options.df_exchange_n_blocks)
    
    the_batch_index = 1
    exchange_batch_indexes = Array{Tuple{Int, Int}}(undef, lower_triangle_length)
    for iii in 1:scf_options.df_exchange_n_blocks
        for jjj in 1:iii
            exchange_batch_indexes[the_batch_index] = (iii, jjj)
            the_batch_index+=1
        end
    end
    scf_data.screening_data.exchange_batch_indexes = exchange_batch_indexes


    if scf_options.contraction_mode != SCF_Keywords.ContractionMode.screened && 
        scf_options.contraction_mode != SCF_Keywords.ContractionMode.default
        return #below setup is not necessary
    end

    #CPU only setup
    scf_data.k_blocks = zeros(Float64, K_block_width, K_block_width, lower_triangle_length)

    total_non_screened_indices = 0
    blocks_to_calculate = Vector{Int}(undef, 0)
    block_index = 1
    block_screen_matrix = zeros(Bool, scf_options.df_exchange_n_blocks, scf_options.df_exchange_n_blocks)

    while block_index <= lower_triangle_length
        pp, qq = exchange_batch_indexes[block_index]
        if scf_options.df_screen_exchange     
            p_range = (pp-1)*K_block_width+1:pp*K_block_width
            q_range = (qq-1)*K_block_width+1:qq*K_block_width
            total_non_screened_indices = sum(
                view(scf_data.screening_data.basis_function_screen_matrix, p_range, q_range))
            if total_non_screened_indices != 0 #skip where all are screened
                push!(blocks_to_calculate, block_index) 
                block_screen_matrix[pp, qq] = true
            end
        else
            push!(blocks_to_calculate, block_index) 
            block_screen_matrix[pp, qq] = true
        end
        block_index += 1
    end
    
    scf_data.screening_data.block_screen_matrix = block_screen_matrix
    scf_data.screening_data.blocks_to_calculate = blocks_to_calculate

    jc_timing.non_timing_data[JCTC.unscreened_exchange_blocks] = string(length(blocks_to_calculate))

   
end

function calculate_K_lower_diagonal_block(scf_data, scf_options)

    W = scf_data.D_tilde
    p = scf_data.μ
    Q = size(W, 1)
    occ = scf_data.occ

    K_block_width = scf_data.screening_data.K_block_width
    
    transA = true
    transB = false
    alpha = -1.0
    beta = 0.0
    linear_indices = LinearIndices(W)

    M = K_block_width
    N = K_block_width
    K = Q * occ
   
    n_threads = Threads.nthreads()
    n_threads = min(n_threads, length(scf_data.screening_data.blocks_to_calculate))
    
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    dynamic_index = n_threads + 1
    dynamic_lock = Threads.ReentrantLock()
    
    exchange_blocks = scf_data.k_blocks

    K_linear_indices = LinearIndices(exchange_blocks)

    Threads.@sync for thread in 1:n_threads
        Threads.@spawn begin
            index = thread
            for ii in thread:n_threads:length(scf_data.screening_data.blocks_to_calculate)
                index = scf_data.screening_data.blocks_to_calculate[ii]
                pp, qq = scf_data.screening_data.exchange_batch_indexes[index]

                    p_range = (pp-1)*K_block_width+1:pp*K_block_width
                    p_start = (pp - 1) * K_block_width + 1

                    q_range = (qq-1)*K_block_width+1:qq*K_block_width
                    q_start = (qq - 1) * K_block_width + 1

                    A_ptr = pointer(W, linear_indices[1, 1, p_start])
                    B_ptr = pointer(W, linear_indices[1, 1, q_start])
                    C_ptr = pointer(exchange_blocks, K_linear_indices[1, 1, thread])


                    call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)

                    scf_data.two_electron_fock[p_range, q_range] .= view(exchange_blocks, :,:, thread)
                    if pp != qq
                        scf_data.two_electron_fock[q_range, p_range] .= transpose(view(exchange_blocks, :,:, thread)) 
                    end
            end
        end
    end#sync
      
    BLAS.set_num_threads(blas_threads)
    if p % scf_options.df_exchange_n_blocks == 0 # square blocks cover the entire pq space
        return
    end
    # non square part of K not include in the blocks above if any are non screened put back after full block is working
    p_non_square_range = 1:p
    p_non_square_start = 1

    #non square part 
    q_nonsquare_range = p-(p%scf_options.df_exchange_n_blocks)+1:p
    q_nonsquare_start = q_nonsquare_range[1]
    M = p
    N = length(q_nonsquare_range)
    K = Q * occ
    

    A_nonsquare_ptr = pointer(W, linear_indices[1, 1, p_non_square_start])
    B_nonsquare_ptr = pointer(W, linear_indices[1, 1, q_nonsquare_start])
    C_nonsquare_ptr = pointer(scf_data.k_blocks, 1)


    call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_nonsquare_ptr, B_nonsquare_ptr, beta, C_nonsquare_ptr)    

    non_square_buffer = reshape(view(scf_data.k_blocks, 1:M*N), (p, length(q_nonsquare_range))) 

    scf_data.two_electron_fock[p_non_square_range, q_nonsquare_range] .= non_square_buffer
    scf_data.two_electron_fock[q_nonsquare_range, p_non_square_range] .= transpose(non_square_buffer)
end

#todo remove this function after paper is published
function calculate_K_lower_diagonal_block_no_screen(scf_data, scf_options)

    W = scf_data.D_tilde
    p = scf_data.μ
    Q = size(W, 1)
    occ = scf_data.occ

    K_block_width = scf_data.screening_data.K_block_width
    
    transA = true
    transB = false
    alpha = -1.0
    beta = 0.0
    linear_indices = LinearIndices(W)

    M = K_block_width
    N = K_block_width
    K = Q * occ
   
    n_threads = Threads.nthreads()
    
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    dynamic_index = n_threads + 1
    dynamic_lock = Threads.ReentrantLock()
    
    exchange_blocks = scf_data.k_blocks

    K_linear_indices = LinearIndices(exchange_blocks)
    lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_n_blocks)

    Threads.@threads for index in 1:lower_triangle_length
        # Threads.@spawn begin
            # index = thread
            # while index <= lower_triangle_length
                pp, qq = scf_data.screening_data.exchange_batch_indexes[index]

                    p_range = (pp-1)*K_block_width+1:pp*K_block_width
                    p_start = (pp - 1) * K_block_width + 1

                    q_range = (qq-1)*K_block_width+1:qq*K_block_width
                    q_start = (qq - 1) * K_block_width + 1

                    A_ptr = pointer(W, linear_indices[1, 1, p_start])
                    B_ptr = pointer(W, linear_indices[1, 1, q_start])
                    C_ptr = pointer(exchange_blocks, K_linear_indices[1, 1, index])


                    call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)

                    scf_data.two_electron_fock[p_range, q_range] .= view(exchange_blocks, :,:, index)
                    if pp != qq
                        scf_data.two_electron_fock[q_range, p_range] .= transpose(view(exchange_blocks, :,:, index)) 
                    end
                # lock(dynamic_lock) do
                #     if dynamic_index <= lower_triangle_length
                #         index = dynamic_index
                #         dynamic_index += 1
                #     else
                #         index = lower_triangle_length + 1
                #     end
                # end
            # end
        # end
    end#sync
      
    BLAS.set_num_threads(blas_threads)
    if p % scf_options.df_exchange_n_blocks == 0 # square blocks cover the entire pq space
        return
    end
    # non square part of K not include in the blocks above if any are non screened put back after full block is working
    p_non_square_range = 1:p
    p_non_square_start = 1

    #non square part 
    q_nonsquare_range = p-(p%scf_options.df_exchange_n_blocks)+1:p
    q_nonsquare_start = q_nonsquare_range[1]
    M = p
    N = length(q_nonsquare_range)
    K = Q * occ
    

    A_nonsquare_ptr = pointer(W, linear_indices[1, 1, p_non_square_start])
    B_nonsquare_ptr = pointer(W, linear_indices[1, 1, q_nonsquare_start])
    C_nonsquare_ptr = pointer(scf_data.k_blocks, 1)


    non_square_gemm_time = @elapsed call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_nonsquare_ptr, B_nonsquare_ptr, beta, C_nonsquare_ptr)    

    println("non square gemm time: ", non_square_gemm_time)

    non_square_buffer = reshape(view(scf_data.k_blocks, 1:M*N), (p, length(q_nonsquare_range))) 

    scf_data.two_electron_fock[p_non_square_range, q_nonsquare_range] .= non_square_buffer
    scf_data.two_electron_fock[q_nonsquare_range, p_non_square_range] .= transpose(non_square_buffer)
end


#todo move this to a BLAS shared file
function call_gemm!(transA::Val, transB::Val,
    M::Int, N::Int, K::Int,
    alpha::Float64, A::Ptr{Float64}, B::Ptr{Float64},
    beta::Float64, C::Ptr{Float64})

    # Convert our compile-time transpose marker to a char for BLAS
    convtrans(V::Val{false}) = 'N'
    convtrans(V::Val{true}) = 'T'

    if transA == Val(false)
        lda = M
    else
        lda = K
    end
    if transB == Val(false)
        ldb = K
    else
        ldb = N
    end
    ldc = M

    ccall((:dgemm_64_, BLAS.libblas), Nothing,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ref{BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
            Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
            Ref{BlasInt}),
        convtrans(transA), convtrans(transB), M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc)
end
