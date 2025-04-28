using MPI
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using LinearAlgebra
using Base.Threads
using HDF5
using JuliaChem.Shared.JCTC
using JuliaChem.Shared.Constants.SCF_Keywords

function df_rhf_fock_build_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions, H::Array{Float64},
    jc_timing::JCTiming) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    p = scf_data.μ
    n_ooc = scf_data.occ


    #get environment variable for number of devices to use
    num_devices = scf_options.num_devices
    scf_data.gpu_data.number_of_devices_used = num_devices

    # num_devices = Int64(length(devices))
    num_devices_global = num_devices*n_ranks  
    occupied_orbital_coefficients = permutedims(occupied_orbital_coefficients, (2,1))


    # n_j_streams_per_device = 20 # put back if J_sym is reimplemented 

    three_center_integrals = Array{Array{Float64}}(undef, num_devices)
    use_K_rect = scf_options.df_use_K_sym && scf_options.df_K_sym_type == SCF_Keywords.GPUAlgorithms.df_K_sym_rect
    if iteration == 1

        if num_devices > length(CUDA.devices())
            error("Number of devices requested is greater than the number of devices available")
        end

        two_eri_time = @elapsed two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
        screening_time = @elapsed begin
            get_screening_metadata!(scf_data, scf_options.df_screening_sigma, jeri_engine_thread, two_center_integrals, basis_sets, jc_timing)
            calculate_exchange_block_screen_matrix(scf_data, scf_options, 
            SCF_Keywords.Screening.df_exchange_n_blocks_gpu_default,
            jc_timing)
        end
        
        three_eri_time = @elapsed begin
            for device_id in 1:num_devices #the method being called uses many threads do not need to thread by device
                global_device_id = device_id + (rank)*num_devices
                three_center_integrals[device_id] = calculate_three_center_integrals(jeri_engine_thread_df,
                     basis_sets, scf_options, scf_data, global_device_id-1,num_devices_global, true)
            end
        end

        calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices, num_devices_global, basis_sets, jc_timing)
       
        if scf_options.df_use_adaptive && scf_options.df_exchange_n_blocks == 0
            QQ = maximum(scf_data.gpu_data.device_Q_range_lengths)

            block_size_deterimined = false
            scf_options.df_exchange_n_blocks = 1
            while !block_size_deterimined && df_exchange_n_blocks < scf_options.df_max_num_GPU_exchange_blocks
                number_of_operations = 2*QQ*n_ooc*(p÷scf_options.df_exchange_n_blocks)^2 
                # println("number of operations per block = ", number_of_operations)
                if number_of_operations > Int64(scf_options.df_GPU_K_block_opeartions_threshold)
                    scf_options.df_exchange_n_blocks += 1
                else # don't go over 16^10 operations per block
                    block_size_deterimined = true
                end
            end
        end

        if scf_options.df_use_K_sym && scf_options.df_exchange_n_blocks < 2
            scf_options.df_exchange_n_blocks = 2
        end

        scf_data.screening_data.K_block_width = p÷scf_options.df_exchange_n_blocks

        #clear the memory 
        two_center_integrals = nothing
        three_center_integrals = nothing

        scf_data.gpu_data.device_fock = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_coulomb_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_coulomb = Array{CuArray{Float64}}(undef, num_devices)


        scf_data.gpu_data.device_exchange_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_occupied_orbital_coefficients = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_density = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_screened_density = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_non_zero_coefficients = Array{Array{CuArray{Float64}}}(undef, num_devices)
        scf_data.gpu_data.device_K_block = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_non_square_K_block = Array{CuArray{Float64}}(undef, num_devices)

        scf_data.gpu_data.device_range_p = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_range_start = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_range_end = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_range_sparse_start = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_range_sparse_end = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_sparse_to_p = Array{CuArray{Int64,1}}(undef, num_devices)
        scf_data.gpu_data.device_sparse_to_q = Array{CuArray{Int64,1}}(undef, num_devices)

        scf_data.gpu_data.sparse_pq_index_map = Array{CuArray{Int64,2}}(undef, num_devices)
        
        scf_data.gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)
        scf_data.density = zeros(Float64, (scf_data.μ,scf_data.μ ))

        scf_data.non_zero_coefficients = zeros(Float64, n_ooc, p, p)

     
        
        scf_data.gpu_data.W_pointers_B = Array{Array{CuPtr{Float64}}}(undef, num_devices)
        scf_data.gpu_data.W_pointers_non_zero_coeff = Array{Array{CuPtr{Float64}}}(undef, num_devices)
        scf_data.gpu_data.W_pointers_W = Array{Array{CuPtr{Float64}}}(undef, num_devices)
        scf_data.gpu_data.W_group_sizes = Array{Array{Int,1}}(undef, num_devices)
        scf_data.gpu_data.W_group_count = Array{Int,1}(undef, num_devices)
        scf_data.gpu_data.W_non_screened_p_indices_count = Array{Array{Int,1}}(undef, num_devices)
        Threads.@threads for device_id in 1:num_devices
            CUDA.device!(device_id-1)

            #device host data 
            global_device_id = device_id + (rank)*num_devices
            Q = scf_data.gpu_data.device_Q_range_lengths[global_device_id]

            #host density until I can figure out how to write a kernel for copying to the screened vector on the gpu
            scf_data.density_array = zeros(Float64, (scf_data.screening_data.screened_indices_count))
            #host fock for transfering in parallel from the GPUs
            scf_data.gpu_data.host_fock[device_id] = zeros(Float64, (scf_data.μ, scf_data.μ))

            #cuda device data
            scf_data.gpu_data.device_fock[device_id] = CUDA.zeros(Float64,(scf_data.μ, scf_data.μ))

            scf_data.gpu_data.device_density[device_id] = CUDA.zeros(Float64, (p,p))
            scf_data.gpu_data.device_screened_density[device_id] = CUDA.zeros(Float64, (scf_data.screening_data.screened_indices_count))
            scf_data.gpu_data.device_coulomb[device_id] = CUDA.zeros(Float64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.device_coulomb_intermediate[device_id] = CUDA.zeros(Float64,(Q))
        

            scf_data.gpu_data.device_occupied_orbital_coefficients[device_id] = CUDA.zeros(Float64, (scf_data.occ, scf_data.μ))
            scf_data.gpu_data.device_non_zero_coefficients[device_id] = CUDA.zeros(Float64, n_ooc, p, p)
            scf_data.gpu_data.device_exchange_intermediate[device_id] =  CUDA.zeros(Float64, (Q, n_ooc, p))
            scf_data.lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_n_blocks)#should only be done on first iteration 
            scf_data.gpu_data.device_K_block[device_id] = CUDA.zeros(Float64, (scf_data.screening_data.K_block_width, scf_data.screening_data.K_block_width, scf_data.lower_triangle_length))
            
            ################   duplicated logic! move this to a shared place   ##########################
            row_nonsquare_range = p-(p%scf_options.df_exchange_n_blocks)+1:p
            scf_data.gpu_data.device_non_square_K_block[device_id] = CUDA.zeros(Float64, (length(row_nonsquare_range), p))
            ############################################################################################################
            
            if rank == 0 && device_id == 1
                scf_data.gpu_data.device_H = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))
                CUDA.copyto!(scf_data.gpu_data.device_H, H)
            end
            #timing gpu size in MB 

            #pointers for W calculation 
            # W_indicies = LinearIndices(size(scf_data.gpu_data.device_exchange_intermediate[device_id])) 
            # B_indicies = LinearIndices(size(scf_data.gpu_data.device_B[device_id])) 
            # non_zero_coeff_indicies = LinearIndices(size(scf_data.gpu_data.device_non_zero_coefficients[device_id]))
        
            scf_data.gpu_data.W_pointers_B[device_id] = Array{CuPtr{Float64}}(undef, scf_data.μ)
            scf_data.gpu_data.W_pointers_non_zero_coeff[device_id] = Array{CuPtr{Float64}}(undef, scf_data.μ)
            scf_data.gpu_data.W_pointers_W[device_id] = Array{CuPtr{Float64}}(undef, scf_data.μ)
          

            #get a list of tuples from the indices of scf_data.screening_data.non_screened_p_indices_count index and value 
            key_value_pairs = [(k, v) for (k, v) in enumerate(scf_data.screening_data.non_screened_p_indices_count)]
            sorted_key_value_pairs = sort(key_value_pairs, by=x->x[2])
            
            scf_data.gpu_data.W_group_sizes[device_id] = Vector{Int64}()
            scf_data.gpu_data.W_non_screened_p_indices_count[device_id] = Vector{Int64}()

            sorted_indices_count_p_tuples = order_gemm_groups_index_count(scf_data.screening_data.non_screened_p_indices_count)
            number_of_groups = 0
            indicies_count = 0
            ordered_index = 1
            #put together pointers for the calculation of W in order by the size of the number of non screened p indices (which is the k dimension of the gemm)
            for k_p_tuple in sorted_indices_count_p_tuples
                # for pp in 1:p
                if indicies_count != k_p_tuple[1] #new group 
                    number_of_groups += 1
                    indicies_count = k_p_tuple[1]
                    push!(scf_data.gpu_data.W_non_screened_p_indices_count[device_id], indicies_count)
                    push!(scf_data.gpu_data.W_group_sizes[device_id], 1)
                else
                    scf_data.gpu_data.W_group_sizes[device_id][number_of_groups] += 1
                end
        
                pp = k_p_tuple[2]
                #pointer to the section of the B matrix that is used for this p
                scf_data.gpu_data.W_pointers_B[device_id][ordered_index] = 
                     pointer(view(scf_data.gpu_data.device_B[device_id], :, scf_data.screening_data.sparse_p_start_indices[pp]:scf_data.screening_data.sparse_p_start_indices[pp]+indicies_count-1))
                #pointer to the section of the non zero coefficients that is used for this p
                scf_data.gpu_data.W_pointers_non_zero_coeff[device_id][ordered_index] = 
                    pointer(view(scf_data.gpu_data.device_non_zero_coefficients[device_id], :,1:indicies_count,pp))
                #pointer to the section of the W matrix that is used for this p
                scf_data.gpu_data.W_pointers_W[device_id][ordered_index] = 
                    pointer(view(scf_data.gpu_data.device_exchange_intermediate[device_id], :,:,pp))

                ordered_index += 1
            end

            scf_data.gpu_data.W_group_count[device_id] = number_of_groups
          
        end 
        
        gpu_screening_setup = @elapsed setup_gpu_screening_data!(scf_data, num_devices)


        jc_timing.non_timing_data[JCTC.contraction_algorithm] = "screened gpu"
        jc_timing.timings[JCTC.two_eri_time] = two_eri_time
        jc_timing.timings[JCTC.three_eri_time] = three_eri_time
        jc_timing.timings[JCTC.screening_time] = screening_time
        jc_timing.timings[JCTC.GPU_screening_setup_time] = gpu_screening_setup
        jc_timing.non_timing_data[JCTC.GPU_num_devices] = string(num_devices)
    end

    V_times = zeros(Float64, num_devices)
    J_times = zeros(Float64, num_devices)

    W_times = zeros(Float64, num_devices)
    K_times = zeros(Float64, num_devices)
    gpu_copy_J_time = zeros(Float64, num_devices)
    gpu_copy_sym_time = zeros(Float64, num_devices)
    density_times = zeros(Float64, num_devices)
    gpu_fock_times = zeros(Float64, num_devices)
    non_zero_coeff_times  = zeros(Float64, num_devices)
    H_add_time = 0.0
    


    n_threads = Threads.nthreads()
    threads_per_device = Int64((n_threads - num_devices) ÷ num_devices) 

    total_fock_gpu_time = @elapsed begin 
        Threads.@sync for device_id in 1:num_devices
            Threads.@spawn begin
                gpu_fock_times[device_id] = @elapsed begin 
                    CUDA.device!(device_id-1)
                    global_device_id = device_id + (rank)*num_devices
                    Q_length = scf_data.gpu_data.device_Q_range_lengths[global_device_id]

                    ooc = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
                    density = scf_data.gpu_data.device_screened_density[device_id]
                    B = scf_data.gpu_data.device_B[device_id]
                    V = scf_data.gpu_data.device_coulomb_intermediate[device_id]
                    W = scf_data.gpu_data.device_exchange_intermediate[device_id]
                    J = scf_data.gpu_data.device_coulomb[device_id]
                    fock = scf_data.gpu_data.device_fock[device_id]
                    host_fock = scf_data.gpu_data.host_fock[device_id]
                    
                
                    CUDA.copyto!(ooc, occupied_orbital_coefficients)
                    CUDA.synchronize()

                    non_zero_coeff_times[device_id] = @elapsed form_nozero_coefficient_matrix!(scf_data, device_id)
                    W_times[device_id]  = @elapsed calculate_W_screened_GPU_batched(device_id, scf_data, threads_per_device)
                    if use_K_rect 
                        K_times[device_id]  = @elapsed calculate_K_upper_diagonal_rectangle_blocks(fock, W, Q_length, device_id,
                        scf_data, scf_options, threads_per_device)
                    elseif scf_options.df_exchange_n_blocks > 1 
                        K_times[device_id]  = @elapsed calculate_K_lower_diagonal_block_no_screen_GPU(host_fock, fock, W, Q_length, device_id,
                        scf_data, scf_options, scf_data.lower_triangle_length, threads_per_device)       
                    else
                        K_times[device_id]  = @elapsed calcululate_K_no_sym_GPU!(fock, W, p, scf_data.occ, Q_length, device_id)
                    end
                    if rank == 0 && device_id == 1
                        H_add_time = @elapsed begin
                            CUDA.axpy!(1.0, scf_data.gpu_data.device_H, fock)
                            CUDA.synchronize()
                        end
                    end

                    density_times[device_id]  = @elapsed form_screened_density!(scf_data, device_id)
                    V_times[device_id]  = @elapsed calculate_V_screened_GPU(V, B, density)
                    J_times[device_id]  = @elapsed calculate_J_screened_GPU(J, B, V)
                    gpu_copy_J_time[device_id] = @elapsed begin 
                        numblocks = ceil(Int64, scf_data.screening_data.screened_indices_count/256)
                        threads = min(256, scf_data.screening_data.screened_indices_count)
    
                        @cuda threads=threads blocks=numblocks copy_screened_J_to_fock_upper_triangle(fock, J, scf_data.gpu_data.device_sparse_to_p[device_id], 
                            scf_data.gpu_data.device_sparse_to_q[device_id], scf_data.screening_data.screened_indices_count)
                        CUDA.synchronize() 
                    end
                  
                    gpu_copy_sym_time[device_id] = @elapsed begin
                        numblocks = ceil(Int64, scf_data.screening_data.screened_indices_count/256)
                        threads = min(256, scf_data.screening_data.screened_indices_count)
    
                        if !use_K_rect 
                            @cuda threads=threads blocks=numblocks copy_upper_to_lower_kernel(fock)
                        else
                            @cuda threads=threads blocks=numblocks copy_lower_to_upper_kernel(fock)
                        end
                        CUDA.synchronize() 
                    end
                end # gpu fock time elapsed
            end #spawn     
        end#sync
    end# total fock gpu time elapsed

    fock_copy_time = @elapsed begin
        Threads.@threads for device_id in 1:num_devices
            CUDA.device!(device_id-1)
            CUDA.copyto!(scf_data.gpu_data.host_fock[device_id], scf_data.gpu_data.device_fock[device_id])  
            CUDA.synchronize()
        end
        scf_data.two_electron_fock = scf_data.gpu_data.host_fock[1]
        for device_id in 2:num_devices
            axpy!(1.0, scf_data.gpu_data.host_fock[device_id], scf_data.two_electron_fock)
        end 
    end #copy_time elapsed 


    for device_id in 1:num_devices
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_W_time, device_id, iteration)] = W_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_V_time, device_id, iteration)] = V_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_J_time, device_id, iteration)] = J_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_K_time, device_id, iteration)] = K_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_density_time, device_id, iteration)] = density_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.gpu_fock_time, device_id, iteration)] = gpu_fock_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_non_zero_coeff_time, device_id, iteration)] = non_zero_coeff_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.gpu_copy_J_time, device_id, iteration)] = gpu_copy_J_time[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.gpu_copy_sym_time, device_id, iteration)] = gpu_copy_sym_time[device_id]
        jc_timing.non_timing_data[JCTiming_GPUkey(JCTC.GPU_data_size_MB, device_id, iteration)] = string(calculate_screened_GPU_data_size_MB(scf_data, device_id))
    end

    jc_timing.timings[JCTiming_key(JCTC.K_time, iteration)] = maximum(K_times)
    jc_timing.timings[JCTiming_key(JCTC.W_time, iteration)] = maximum(W_times)
    jc_timing.timings[JCTiming_key(JCTC.V_time, iteration)] = maximum(V_times)
    jc_timing.timings[JCTiming_key(JCTC.J_time, iteration)] = maximum(J_times)
    jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_H_add_time, 1, iteration)] = H_add_time


    jc_timing.timings[JCTiming_key(JCTC.fock_gpu_cpu_copy_reduce_time, iteration)] = fock_copy_time
    jc_timing.timings[JCTiming_key(JCTC.total_fock_gpu_time, iteration)] = total_fock_gpu_time

    println("max W time = ", maximum(W_times))
    println("max K time = ", maximum(K_times))
    println("total fock time = ", total_fock_gpu_time + fock_copy_time)
    # gc_time = @elapsed GC.gc()
    # println("gc time = ", gc_time)
end


#to remove branching I need a map from screened[1d index] to unscreened 2d[p,q] indices 
#not a huge performance hit at the moment so not proritiezed 
function form_screened_density_kernel!(screened_density::CuDeviceArray{Float64}, density::CuDeviceArray{Float64}, 
    sparse_pq_index_map::CuDeviceArray{Int64}, p::Int64)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for pp in index:stride:p
        for qq in 1:pp-1
            if sparse_pq_index_map[pp, qq] == 0
                continue
            else 
                @inbounds screened_density[sparse_pq_index_map[pp, qq]] = 2.0*density[pp, qq] # symmetric multiplication 2.0* for off diagonal
            end
        end
        @inbounds screened_density[sparse_pq_index_map[pp, pp]] = density[pp, pp]  # and 1.0* for diagonal
    end  
end

function form_screened_density!(scf_data::SCFData, device_id::Int64)
    p = scf_data.μ
    density = scf_data.gpu_data.device_density[device_id]

    screened_density = scf_data.gpu_data.device_screened_density[device_id]
    occupied_orbital_coefficients = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]

    CUDA.CUBLAS.gemm!('T', 'N', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, density)
    CUDA.synchronize()
    
    sparse_pq_index_map = scf_data.gpu_data.sparse_pq_index_map[device_id]

    numblocks = ceil(Int64, p/256)
    threads = min(256, p)

    @cuda threads=threads blocks=numblocks form_screened_density_kernel!(screened_density, density, sparse_pq_index_map, p)
    CUDA.synchronize()

end

function setup_gpu_screening_data!(scf_data::SCFData, num_devices::Int64)
    n_ranges = 0
    p = scf_data.μ
    n_ranges_arr = zeros(Int64, p)
    Threads.@threads for pp in 1:p
        n_ranges_arr[pp] = length(scf_data.screening_data.non_zero_ranges[pp])
    end
    n_ranges = sum(n_ranges_arr)

    p_non_zero_ranges_start = zeros(Int64, p) #the nth range that corresponds to the first range for p 
    p_non_zero_ranges_start[1] = 1
    for pp in 2:p
        p_non_zero_ranges_start[pp] = p_non_zero_ranges_start[pp-1] + n_ranges_arr[pp-1]
    end
    scf_data.gpu_data.n_screened_occupied_orbital_ranges = n_ranges

    range_p = Array{Int64}(undef, n_ranges)
    range_start = Array{Int64}(undef, n_ranges)
    range_end = Array{Int64}(undef, n_ranges)
    range_sparse_start = Array{Int64}(undef, n_ranges)
    range_sparse_end = Array{Int64}(undef, n_ranges)




    range_index = 1
    Threads.@threads for pp in 1:p 
        pp_range_index = 1
        for range_index in p_non_zero_ranges_start[pp]:p_non_zero_ranges_start[pp]+n_ranges_arr[pp]-1
            range_p[range_index] = pp
            range_start[range_index] = scf_data.screening_data.non_zero_ranges[pp][pp_range_index][1]
            range_end[range_index] = scf_data.screening_data.non_zero_ranges[pp][pp_range_index][end]
            range_sparse_start[range_index] = scf_data.screening_data.non_zero_sparse_ranges[pp][pp_range_index][1]
            range_sparse_end[range_index] = scf_data.screening_data.non_zero_sparse_ranges[pp][pp_range_index][end]
            pp_range_index += 1
        end
    end
    
    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            CUDA.device!(device_id-1)
            scf_data.gpu_data.device_range_p[device_id] = CUDA.zeros(Int, n_ranges)
            scf_data.gpu_data.device_range_start[device_id] = CUDA.zeros(Int, n_ranges)
            scf_data.gpu_data.device_range_end[device_id] = CUDA.zeros(Int, n_ranges)
            scf_data.gpu_data.device_range_sparse_start[device_id] = CUDA.zeros(Int, n_ranges)
            scf_data.gpu_data.device_range_sparse_end[device_id] = CUDA.zeros(Int, n_ranges)    
            scf_data.gpu_data.device_sparse_to_p[device_id] = CUDA.zeros(Int64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.device_sparse_to_q[device_id] = CUDA.zeros(Int64, scf_data.screening_data.screened_indices_count)
            
            scf_data.gpu_data.sparse_pq_index_map[device_id] = CUDA.zeros(Int, (p, p))

            CUDA.copyto!(scf_data.gpu_data.sparse_pq_index_map[device_id], scf_data.screening_data.sparse_pq_index_map)
            CUDA.copyto!(scf_data.gpu_data.device_range_p[device_id], range_p)
            CUDA.copyto!(scf_data.gpu_data.device_range_start[device_id], range_start)
            CUDA.copyto!(scf_data.gpu_data.device_range_end[device_id], range_end)
            CUDA.copyto!(scf_data.gpu_data.device_range_sparse_start[device_id], range_sparse_start)
            CUDA.copyto!(scf_data.gpu_data.device_range_sparse_end[device_id], range_sparse_end)
            CUDA.synchronize() 
        end
    end

    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            CUDA.device!(device_id-1)
            numblocks = ceil(Int64, n_ranges/256)
            threads = min(256, n_ranges)

            @cuda threads=threads blocks=numblocks create_sparse_to_p_q_kernel(scf_data.gpu_data.device_sparse_to_p[device_id],
                scf_data.gpu_data.device_sparse_to_q[device_id], 
                scf_data.gpu_data.sparse_pq_index_map[device_id], p)
            CUDA.synchronize()
        end 
    end
end

function create_sparse_to_p_q_kernel(sparse_to_p::CuDeviceArray{Int64}, 
    sparse_to_q::CuDeviceArray{Int64}, 
    sparse_pq_index_map::CuDeviceArray{Int64},  p::Int64)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for pp in index:stride:p
        for qq in 1:p
            sparse_index = sparse_pq_index_map[pp, qq]
            if sparse_index != 0
                sparse_to_p[sparse_index] = pp
                sparse_to_q[sparse_index] = qq
            end
        end
    end

end

function form_nozero_coefficient_matrix!(scf_data::SCFData, device_id :: Int64)
    CUDA.device!(device_id-1)
    

    n_ranges = scf_data.gpu_data.n_screened_occupied_orbital_ranges

    numblocks = ceil(Int64, n_ranges/256)
    threads = min(256, n_ranges)

    @cuda threads=threads blocks=numblocks build_non_zero_coefficients_kernel(scf_data.gpu_data.device_non_zero_coefficients[device_id], 
        scf_data.gpu_data.device_occupied_orbital_coefficients[device_id], 
        scf_data.gpu_data.device_range_p[device_id],
        scf_data.gpu_data.device_range_start[device_id],
        scf_data.gpu_data.device_range_end[device_id],
        scf_data.gpu_data.device_range_sparse_start[device_id],
        n_ranges)
    CUDA.synchronize()
end

function build_non_zero_coefficients_kernel(non_zero_coefficients::CuDeviceArray{Float64}, 
     occupied_orbital_coefficients::CuDeviceArray{Float64},
        device_range_p::CuDeviceArray{Int64},
        device_range_start::CuDeviceArray{Int64},
        device_range_end::CuDeviceArray{Int64},
        device_range_sparse_start::CuDeviceArray{Int64},
        n_ranges::Int64)

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:n_ranges
        pp = device_range_p[i]
        range_start = device_range_start[i]
        range_end = device_range_end[i]
        range_sparse_start = device_range_sparse_start[i]
        # range_sparse_end = device_range_sparse_end[i]

        for j in 0:range_end-range_start
            non_zero_coefficients[:, range_sparse_start+j, pp] .= view(occupied_orbital_coefficients, :, range_start+j)
        end
    end
end

function copy_screened_J_to_fock(fock::CuDeviceArray{Float64}, J::CuDeviceArray{Float64},
        device_sparse_to_p::CuDeviceArray{Int64}, device_sparse_to_q::CuDeviceArray{Int64}, n_sparse_indicies::Int64)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    
    for i = index:stride:n_sparse_indicies
        pp = device_sparse_to_p[i]
        qq = device_sparse_to_q[i]
        @inbounds fock[pp, qq] +=J[i]
    end
    return
end


function copy_screened_J_to_fock_upper_triangle(fock::CuDeviceArray{Float64}, J::CuDeviceArray{Float64},
    device_sparse_to_p::CuDeviceArray{Int64}, device_sparse_to_q::CuDeviceArray{Int64}, n_sparse_indicies::Int64)

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x


    for i = index:stride:n_sparse_indicies
        qq = device_sparse_to_p[i]
        pp = device_sparse_to_q[i]
        @inbounds fock[pp, qq] += J[i]
    end
    return
end

function copy_lower_to_upper_kernel(A ::CuDeviceArray{Float64})
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:size(A, 1)
        for j = axes(A, 2)
            @inbounds A[j, i] = Float64(i > j) * A[i, j] + Float64(i < j)* A[j, i]  + Float64(i == j) * A[i, j] 
            # 1st term if (i,j) is in the lower triangle copy to (j,i) in the upper triangle else do nothing
            # 2nd term if (i,j) is in the upper triangle keep the value the same otherwise do nothing
            # 3rd term if on the diagonal keep the value the same else do nothing
        end
    end
    return
end

function copy_upper_to_lower_kernel(A ::CuDeviceArray{Float64})
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:size(A, 1)
        for j = axes(A, 2)
            @inbounds A[j, i] = A[i, j] 
        end
    end
    return
end


function calculate_V_screened_GPU(V::CuArray, B::CuArray, density::CuArray)
    CUDA.CUBLAS.gemv!('N', 1.0, B, density, 0.0, V)
    CUDA.synchronize()
end

function calculate_J_screened_GPU(J::CuArray,  B::CuArray, V::CuArray)
    CUDA.CUBLAS.gemv!('T', 2.0, B, V, 0.0, J)
    CUDA.synchronize()
end


function calculate_W_screened_GPU(device_id, scf_data::SCFData, num_threads ::Int64)
    p = scf_data.μ
   
    alpha = 1.0
    beta = 0.0

    B = scf_data.gpu_data.device_B[device_id] # B intermediate from integral contraction 
    W = scf_data.gpu_data.device_exchange_intermediate[device_id] # W intermediate for exchange calculation

    non_zero_coefficients = scf_data.gpu_data.device_non_zero_coefficients[device_id]
    num_streams = min(16, num_threads)
    p_per_stream = p÷num_streams

    # Threads.@sync begin 
    #     for stream_id in 1:num_streams
            # Threads.@spawn begin
                # pp_start = (stream_id-1)*p_per_stream + 1
                # pp_end = stream_id*p_per_stream
                # if stream_id == num_streams
                #     pp_end = p
                # end
                CUDA.device!(device_id-1)
                for pp in 1:p
                    K = scf_data.screening_data.non_screened_p_indices_count[pp]
                    A_cu = view(B, :, scf_data.screening_data.sparse_p_start_indices[pp]:
                        scf_data.screening_data.sparse_p_start_indices[pp]+K-1)
                    B_cu = view(non_zero_coefficients, :,1:K,pp)
                    C_cu = view(W, :,:,pp)
                    CUDA.CUBLAS.gemm!('N','T', alpha, A_cu, B_cu, beta, C_cu)
                end
            # end
    #     end 
    # end
    CUDA.synchronize()


end

function pointer_gemm_grouped_batched!(
    transA::Vector{CUDA.CUBLAS.cublasOperation_t},
    transB::Vector{CUDA.CUBLAS.cublasOperation_t},
    alpha_array::Vector{Float64},
    m_array::Vector{Int64},
    n_array::Vector{Int64},
    k_array::Vector{Int64},
    A::Vector{CuPtr{Float64}},
    lda_array::Vector{Int64},
    B::Vector{CuPtr{Float64}},
    ldb_array::Vector{Int64},
    beta_array::Vector{Float64},
    C::Vector{CuPtr{Float64}},
    ldc_array::Vector{Int64},
    group_count::Int64,
    group_size::Vector{Int64})

    Aptrs = CuArray(A)
    Bptrs = CuArray(B)
    Cptrs = CuArray(C)
    hndle = CUDA.CUBLAS.handle()
    try
        ## XXX: cublasXgemmGroupedBatched does not seem to support device pointers
        CUDA.CUBLAS.cublasSetPointerMode_v2(hndle, CUDA.CUBLAS.CUBLAS_POINTER_MODE_HOST)

    if CUBLAS.version() >= v"12.0"
        CUDA.CUBLAS.cublasDgemmGroupedBatched_64(hndle, transA, transB, m_array, n_array, k_array, alpha_array, Aptrs, lda_array,
            Bptrs, ldb_array, beta_array, Cptrs, ldc_array, group_count, group_size)
    else
        CUDA.CUBLAS.cublasDgemmGroupedBatched(hndle, transA, transB, m_array, n_array, k_array, alpha_array, Aptrs, lda_array,
            Bptrs, ldb_array, beta_array, Cptrs, ldc_array, group_count, group_size)
    end
    finally
        CUDA.CUBLAS.cublasSetPointerMode_v2(hndle, CUDA.CUBLAS.CUBLAS_POINTER_MODE_DEVICE)
    end
    CUDA.unsafe_free!(Cptrs)
    CUDA.unsafe_free!(Bptrs)
    CUDA.unsafe_free!(Aptrs)
    
end

function calculate_W_screened_GPU_batched(device_id, scf_data::SCFData, num_threads ::Int64)
   
    #gemm parameters m = lda, n = ldb, k = ldc
    m = scf_data.gpu_data.device_Q_range_lengths[device_id] #device Q length
    n = scf_data.occ 
    group_count = scf_data.gpu_data.W_group_count[device_id]

    CUDA.device!(device_id-1)

    transa_array = Vector{CUDA.CUBLAS.cublasOperation_t}(undef,  group_count)
    transa_array .=  CUDA.CUBLAS.CUBLAS_OP_N
    transb_array = Vector{CUDA.CUBLAS.cublasOperation_t}(undef,  group_count)
    transb_array .= CUDA.CUBLAS.CUBLAS_OP_T

    m_array = zeros(Int64, group_count)
    m_array .= m
    n_array = zeros(Int64, group_count)
    n_array .= n

    alpha_array = zeros(Float64, group_count)
    alpha_array .= 1.0
    beta_array = zeros(Float64, group_count)
    beta_array .= 0.0

   
    k_array = scf_data.gpu_data.W_non_screened_p_indices_count[device_id]

    pointer_gemm_grouped_batched!(transa_array, transb_array, 
    alpha_array, m_array, n_array, k_array,
        scf_data.gpu_data.W_pointers_B[device_id], m_array, 
        scf_data.gpu_data.W_pointers_non_zero_coeff[device_id], n_array,
        beta_array, scf_data.gpu_data.W_pointers_W[device_id], m_array, 
        group_count, scf_data.gpu_data.W_group_sizes[device_id])   
    CUDA.synchronize()

    # transA::Vector{CUDA.CUBLAS.cublasOperation_t},
    # transB::Vector{CUDA.CUBLAS.cublasOperation_t},
    # alpha_array::Vector{Float64},
    # m_array::Vector{Int64},
    # n_array::Vector{Int64},
    # k_array::Vector{Int64},
    # A::Vector{CuPtr{Float64}},
    # lda_array::Vector{Int64},
    # B::Vector{CuPtr{Float64}},
    # ldb_array::Vector{Int64},
    # beta_array::Vector{Float64},
    # C::Vector{CuPtr{Float64}},
    # ldc_array::Vector{Int64},
    # group_count::Int64,
    # group_size::Vector{Int64})

end

function calcululate_K_no_sym_GPU!(fock::CuArray{Float64,2}, W::CuArray{Float64,3},p::Int64, n_ooc::Int64, Q::Int64, device_id::Int64)
    CUDA.CUBLAS.gemm!('T', 'N', -1.0, reshape(W, (Q*n_ooc, p)), reshape(W, (Q*n_ooc, p)), 0.0, fock)
    CUDA.synchronize()
end

function calculate_K_upper_diagonal_rectangle_blocks(fock::CuArray{Float64,2}, W::CuArray{Float64,3}, Q_length::Int, 
    device_id, scf_data::SCFData, scf_options::SCFOptions, num_threads::Int64)
    CUDA.device!(device_id-1)

    n_occ = scf_data.occ
    p = scf_data.μ
    Q = Q_length #device Q length

    transA = 'T'
    transB = 'N'
    alpha = -1.0
    beta = 0.0

    if scf_options.df_exchange_n_blocks == 0
        n_blocks = 2
        if p > 1600
            n_blocks = 16
        elseif p > 800
            n_blocks = 8
        elseif p > 400
            n_blocks = 4
        end
    else
        n_blocks = scf_options.df_exchange_n_blocks
    end

    block_width = p÷n_blocks

    scf_data.gpu_data.device_K_block = Array{Array{CuArray{Float64}}}(undef, 1)
    scf_data.gpu_data.device_K_block[device_id] = Array{CuArray{Float64}}(undef, n_blocks)

    for block_index in 1:n_blocks
        scf_data.gpu_data.device_K_block[device_id][block_index] = CUDA.zeros(Float64, (block_index*block_width, block_width)) #todo move to iteration 1
    end
    device_K_blocks = scf_data.gpu_data.device_K_block[device_id]
    
    num_streams = min(num_threads, n_blocks)

    Threads.@sync begin
        for stream in 1:num_streams
            Threads.@spawn begin
                CUDA.device!(device_id-1)
                for block_index in stream:num_streams:n_blocks
                    M = block_index*block_width
                    block_p_range = 1:M
                    block_q_range = (block_index-1)*block_width+1:block_index*block_width


                    A = reshape(view(W, :,:, block_p_range), (Q*n_occ, M))
                    B = reshape(view(W, :,:, block_q_range), (Q*n_occ, block_width))
                    
                    CUDA.CUBLAS.gemm!(transA, transB, alpha, A, B, beta, device_K_blocks[block_index]) #transpose(W[:,1:M])*W[:,q_start:q_end] = retangular block of size M X block_width 
                    # fock[block_p_range, block_q_range] .= scf_data.gpu_data.device_K_block[block_index]
                    CUDA.synchronize()

                    CUDA.copyto!(view(fock, block_p_range, block_q_range), device_K_blocks[block_index])
                    # CUDA.copyto!(view(fock, block_q_range, block_p_range), transpose(device_K_blocks[block_index]))
                    CUDA.synchronize()
                end
            end 
        end
        if p%n_blocks != 0
            Threads.@spawn begin
                M = p
                N = p%n_blocks
                K = Q*n_occ

                q_non_square_range = p-N+1:p


                A_non_square = reshape(view(W, :,:, 1:p), (K, p))
                B_non_square = reshape(view(W, :,:, q_non_square_range), (K, N))
                C_non_square = view(fock, :, q_non_square_range)

                CUDA.CUBLAS.gemm!(transA, transB, alpha, A_non_square, B_non_square, beta, C_non_square)
            
                CUDA.synchronize()
                # CUDA.copyto!(view(fock, q_non_square_range, :), C_non_square)
            end
        end      
    end
end


function calculate_K_lower_diagonal_block_no_screen_GPU(host_fock::Array{Float64,2},
     fock::CuArray{Float64,2}, W::CuArray{Float64,3}, Q_length::Int, 
     device_id, scf_data::SCFData, scf_options::SCFOptions,
     lower_triangle_length :: Int64, num_threads_avail::Int64)

    CUDA.device!(device_id-1)

    n_ooc = scf_data.occ
    p = scf_data.μ
    Q = Q_length #device Q length
    K_block_width = scf_data.screening_data.K_block_width

    transA = 'T'
    transB = 'N'
    alpha = -1.0
    beta = 0.0

    M = K_block_width
    N = K_block_width
    K = Q * n_ooc



    device_K_block = scf_data.gpu_data.device_K_block[device_id]

    # no streams if the system is large enough to use this method the GEMM should saturate GPU
    CUDA.device!(device_id-1)
    pp, qq = scf_data.screening_data.exchange_batch_indexes[1]
    p_range = 1:2
    q_range = 1:2
    A = reshape(view(W, :,:, p_range), (K, 2))
    B = reshape(view(W, :,:, q_range), (K, 2))
    for index in 1:lower_triangle_length
        exchange_block = view(device_K_block, :,:, index)

        pp, qq = scf_data.screening_data.exchange_batch_indexes[index]
        p_range = (pp-1)*K_block_width+1:pp*K_block_width        
        q_range = (qq-1)*K_block_width+1:qq*K_block_width

        A = reshape(view(W, :,:, p_range), (K, K_block_width))
        B = reshape(view(W, :,:, q_range), (K, K_block_width))

        CUDA.CUBLAS.gemm!(transA, transB, alpha, A, B, beta, exchange_block)
        CUDA.copyto!(view(fock, p_range, q_range), exchange_block)
        #copy transpose 
        CUDA.copyto!(view(fock, q_range, p_range), transpose(exchange_block))
    end
    CUDA.synchronize()
    if p % scf_options.df_exchange_n_blocks != 0 # if square blocks don't cover the entire pq space
        col_non_square_range = 1:p    
        #non square part that didn't fit in blocks
        row_non_square_range = p-(p%scf_options.df_exchange_n_blocks)+1:p
        
        M = length(row_non_square_range)
        N = p
        
        A_non_square = reshape(view(W, :,:, row_non_square_range), (K, M))
        B_non_square = reshape(view(W, :,:, col_non_square_range), (K, N))
        C_non_square = scf_data.gpu_data.device_non_square_K_block[device_id]
        
    
        CUDA.CUBLAS.gemm!(transA, transB, alpha, A_non_square, B_non_square, beta, C_non_square) #W^T[M, Q*n_ooc] * W[Q*n_ooc, N] = C_non_square[M, N]

        CUDA.copyto!(view(fock, row_non_square_range,:), C_non_square)  #non contiguous memory access on the GPU bad, should use the other triangle side
        #copy transpose
        CUDA.copyto!(view(fock, :, row_non_square_range), transpose(C_non_square))
        CUDA.synchronize()
    end 
end

function calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices, num_devices_global, basis_sets, jc_timing)
    COMM = MPI.COMM_WORLD
    rank = MPI.Comm_rank(COMM)
    n_ranks = MPI.Comm_size(COMM)
    pq = scf_data.screening_data.screened_indices_count
    scf_data.gpu_data.device_B = Array{CuArray{Float64}}(undef, num_devices)
    device_three_center_integrals = Array{CuArray{Float64}}(undef, num_devices)
    host_B_send_buffers = Array{Array{Float64}}(undef, num_devices)
    device_J_AB_invt = Array{CuArray{Float64}}(undef, num_devices)
    device_B_send_buffers = Array{CuArray{Float64}}(undef, num_devices)

    device_B = scf_data.gpu_data.device_B


    device_Q_indices, 
    device_rank_Q_indices, 
    device_Q_range_lengths, 
    max_device_Q_range_length  = calculate_device_ranges_GPU(scf_data, num_devices, n_ranks, basis_sets)

    scf_data.gpu_data.device_Q_range_lengths = device_Q_range_lengths
    scf_data.gpu_data.device_Q_indices = device_Q_indices

    device_id_offset = rank * num_devices
    
    Threads.@sync for setup_device_id in 1:num_devices
        Threads.@spawn begin
            CUDA.device!(setup_device_id-1)
            global_device_id = setup_device_id  + device_id_offset
            # buffer for J_AB_invt for each device max size needed is A*A 
            # for certain B calculations the device will only need a subset of this
            # and will ref   device!(device_id - 1)reference it with a view referencing the front of the underlying array
            device_J_AB_invt[setup_device_id] = CUDA.zeros(Float64, (scf_data.A, scf_data.A))

           
            #todo calculate the three center integrals per device (probably could directly copy to the device while it is being calculated)
            device_three_center_integrals[setup_device_id] = CUDA.zeros(Float64, size(three_center_integrals[setup_device_id]))
            CUDA.copyto!(device_three_center_integrals[setup_device_id], three_center_integrals[setup_device_id])

            device_B[setup_device_id] = CUDA.zeros(Float64, (device_Q_range_lengths[global_device_id], pq))
            if num_devices_global > 1 
                device_B_send_buffers[setup_device_id] = CUDA.zeros(Float64, (max_device_Q_range_length * pq))
                host_B_send_buffers[setup_device_id] = zeros(Float64, (max_device_Q_range_length * pq))
            end
            CUDA.synchronize()
        end #spawn
    end

    if rank == 0
        CUDA.device!(0)
        J_AB_time = @elapsed begin
            CUDA.copyto!(device_J_AB_invt[1], two_center_integrals)
            CUDA.synchronize()
            CUDA.CUSOLVER.potrf!('L', device_J_AB_invt[1])
            CUDA.synchronize()
            CUDA.CUSOLVER.trtri!('L', 'N', device_J_AB_invt[1])
            CUDA.synchronize()
        end

        CUDA.copyto!(two_center_integrals, device_J_AB_invt[1]) # copy back because taking subarrays on the GPU is slow / doesn't work. Need to look into if this is possible with CUDA.jl
        
        jc_timing.timings[JCTC.form_J_AB_inv_time] = J_AB_time
    end

    if MPI.Comm_size(COMM) > 1
        #broadcast two_center_integrals to all ranks
        MPI.Bcast!(two_center_integrals, 0, COMM)
    end
    

    if n_ranks == 1 && num_devices == 1
        CUDA.copyto!(device_J_AB_invt[1], two_center_integrals)
        CUDA.synchronize()
        B_time = @elapsed begin
            CUDA.CUBLAS.trmm!('L', 'L', 'N', 'N', 1.0, device_J_AB_invt[1], device_three_center_integrals[1], device_B[1])   
            CUDA.synchronize() 
        end
        CUDA.unsafe_free!(device_J_AB_invt[1])
        CUDA.unsafe_free!(device_three_center_integrals[1])
        CUDA.reclaim()
        jc_timing.timings[JCTC.B_time] = B_time
        return
    end

    for device_id_two_eri in 2:num_devices
        CUDA.device!(device_id_two_eri-1)
        CUDA.copyto!(device_J_AB_invt[device_id_two_eri], two_center_integrals)
        CUDA.synchronize()
    end

    
    B_time = @elapsed begin
        for global_recieve_device_id in 1:num_devices_global
            rec_device_Q_range_length = device_Q_range_lengths[global_recieve_device_id]
            recieve_rank = (global_recieve_device_id-1) ÷ num_devices
            rank_recieve_device_id = ((global_recieve_device_id-1) % num_devices) + 1 # one indexed device id for the rank 
            array_size = rec_device_Q_range_length*pq
            Threads.@sync for r_send_device_id in 1:num_devices
                Threads.@spawn begin
                    CUDA.device!(r_send_device_id-1) 
                        global_send_device_id = r_send_device_id + device_id_offset 
                        send_device_Q_range_length = device_Q_range_lengths[global_send_device_id]
                        J_AB_invt_for_device = two_center_integrals[device_Q_indices[global_recieve_device_id],device_Q_indices[global_send_device_id]]
                        device_J_AB_inv_count = send_device_Q_range_length*rec_device_Q_range_length # total number of elements in the J_AB_invt matrix for the device
                        CUDA.copyto!(device_J_AB_invt[r_send_device_id],1,J_AB_invt_for_device,1,device_J_AB_inv_count) #copy the needed J_AB_invt data to the device 

                        J_AB_INV_view = reshape(
                            view(device_J_AB_invt[r_send_device_id],
                            1:device_J_AB_inv_count), (rec_device_Q_range_length,send_device_Q_range_length))


                        if global_send_device_id == global_recieve_device_id
                            CUDA.CUBLAS.gemm!('N', 'N', 1.0, J_AB_INV_view, device_three_center_integrals[r_send_device_id], 1.0, device_B[rank_recieve_device_id])
                        else
                            send_B_view = view(device_B_send_buffers[r_send_device_id], 1:array_size)
                            CUDA.CUBLAS.gemm!('N', 'N', 1.0, J_AB_INV_view, device_three_center_integrals[r_send_device_id],
                                0.0, reshape(send_B_view, (rec_device_Q_range_length, pq)))
                        end
                        CUDA.synchronize()

                end #spawn
            end #sync for 

            for send_rank in 0:n_ranks-1
                for rank_send_device_id in 1:num_devices
                    
                    global_send_device_id = num_devices*send_rank + rank_send_device_id
                
                    # skip if the device is the same as the recieve device or if rank is not the sender or reciever
                    if global_send_device_id == global_recieve_device_id || (rank != recieve_rank && rank != send_rank)
                        continue
                    end

                    #copy from the sending device to the host send buffer
                    CUDA.device!(rank_send_device_id-1) do 
                        CUDA.copyto!(host_B_send_buffers[rank_send_device_id], 1, 
                        device_B_send_buffers[rank_send_device_id], 1, array_size)
                    end
                    

                    send_unique_tag = send_rank*100000 + recieve_rank*10000 + global_send_device_id*1000 + global_recieve_device_id*100
                    if send_rank == recieve_rank # devices belong to the same rank 
                        CUDA.device!(rank_recieve_device_id-1) do 
                        CUDA.copyto!(device_B_send_buffers[rank_recieve_device_id],
                            1, host_B_send_buffers[rank_send_device_id], 1, array_size)
                        end
                        
                    elseif rank == send_rank
                        CUDA.device!(rank_send_device_id-1) do 
                            MPI.Send(host_B_send_buffers[rank_send_device_id], recieve_rank, send_unique_tag , COMM)
                        end
                    elseif rank == recieve_rank
                        CUDA.device!(rank_recieve_device_id-1) do 
                            MPI.Recv!(host_B_send_buffers[rank_recieve_device_id], send_rank, send_unique_tag, COMM)
                            CUDA.copyto!(device_B_send_buffers[rank_recieve_device_id],1,
                            host_B_send_buffers[rank_recieve_device_id],1, array_size)
                            CUDA.synchronize()
                        end
                    end
                        
                    if rank == recieve_rank # add the sent buffer to the device B matrix
                        CUDA.device!(rank_recieve_device_id-1) do 
                            device_B_send_view = reshape(view(device_B_send_buffers[rank_recieve_device_id], 1:array_size), (rec_device_Q_range_length, pq))
                            CUDA.axpy!(1.0, device_B_send_view, device_B[rank_recieve_device_id])
                            CUDA.synchronize()
                        end                
                    end
                end
            end
        end
    end

    jc_timing.timings[JCTC.B_time] = B_time

    for device_id in 1:num_devices
        CUDA.device!(device_id-1)
        CUDA.unsafe_free!(device_B_send_buffers[device_id])
        CUDA.unsafe_free!(device_J_AB_invt[device_id])
        CUDA.unsafe_free!(device_three_center_integrals[device_id])
    end

end


# get the index data when using GPUs
# indexes go from rank 0 GPUs in order to rank N gpus in order
# E.g. if 4 GPUs per rank and 4 ranks 
# global device ID: 1, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
# rank device ID:   1, 2, 3, 4, 1, 2, 3, 4, 1, 2,  3,  4,  1,  2,  3,  4
# rank:             0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  2,  2,  3,  3,  3,  3
# with scf_data.A = 63
# ranges[1] = 1:4
# ranges[2] = 5:8
# ...
# ranges[16] = 61:63
# Currently we assume easch rank has the same number of devices
# A GPU device behaves as though it were a CPU rank in non-GPU mode
# uses the indicies as they are calculated by the static_load_rank_indicies_3_eri function
# which is used for the 3-eri load balancing to ranks and devices
function calculate_device_ranges_GPU(scf_data, num_devices, n_ranks, basis_sets)

    load_balance_indicies = []
    total_devices = num_devices*n_ranks
    for rank in 0:total_devices-1
        push!(load_balance_indicies, static_load_rank_indicies_3_eri(rank, total_devices, basis_sets))
    end

    device_Q_indices = Array{UnitRange{Int64}}(undef, num_devices * n_ranks)
    device_rank_Q_indices = Array{UnitRange{Int64}}(undef, num_devices * n_ranks)
    indices_per_device = Array{Int}(undef, num_devices * n_ranks)

    global_device_id = 1

    max_device_Q_range_length = 0


    for rank in 0:n_ranks-1
        for device in 1:num_devices
            device_basis_indicies = load_balance_indicies[global_device_id][2]
            device_Q_indices[global_device_id] = device_basis_indicies[1]:device_basis_indicies[end]
            indices_per_device[global_device_id] = length(device_Q_indices[global_device_id])
            device_rank_Q_indices[global_device_id] = 1:indices_per_device[global_device_id]
            if indices_per_device[global_device_id] > max_device_Q_range_length
                max_device_Q_range_length = indices_per_device[global_device_id]
            end
            global_device_id += 1
        end
    end
    return device_Q_indices, device_rank_Q_indices, indices_per_device, max_device_Q_range_length
end

function calculate_screened_GPU_data_size_MB(scf_data::SCFData, device_id)
    gpu_data_size_MB = 0.0

    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_B[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_coulomb_intermediate[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_exchange_intermediate[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_occupied_orbital_coefficients[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_density[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_fock[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_H)
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_Q_range_lengths[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_Q_indices[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_K_block[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_non_square_K_block[device_id])
    
    return gpu_data_size_MB / 1024^2
end


function order_gemm_groups_index_count(non_screened_p_indices_count)
    k_p_tuples = []
    k_p_dict = Dict{Int64, Vector{Int64}}() #mapping between non_screen_p_indicies_count and index p 
    for (pp, k) in enumerate(non_screened_p_indices_count)
        push!(k_p_tuples, (k, pp))
    end
    # length_of_groups = length.(values(k_p_dict))
    # println("length_of_groups: ", length_of_groups)
    # Sort the tuples by k
    sorted_tuples = sort(k_p_tuples, by=x -> x[1], rev=true)
    return sorted_tuples
end