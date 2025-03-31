using MPI
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER

using AMDGPU
import AMDGPU.rocSOLVER: potrf!
import AMDGPU.rocBLAS: trmm!, gemm!, gemv!

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

    F_ARR_type, I_ARR_type = get_array_types(scf_data.gpu_data.GPU_Type)


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

        if num_devices > GPU_num_devices(scf_data.gpu_data.GPU_Type)
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

        calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices, num_devices_global, basis_sets, jc_timing, F_ARR_type)
       
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
        
        scf_data.density = zeros(Float64, (scf_data.μ,scf_data.μ ))
        scf_data.non_zero_coefficients = zeros(Float64, n_ooc, p, p)

        
        Threads.@threads for device_id in 1:num_devices
            set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)

            #device host data 
            global_device_id = device_id + (rank)*num_devices
            Q = scf_data.gpu_data.device_Q_range_lengths[global_device_id]

            #host density until I can figure out how to write a kernel for copying to the screened vector on the gpu
            scf_data.density_array = zeros(Float64, (scf_data.screening_data.screened_indices_count))
            #host fock for transfering in parallel from the GPUs
            scf_data.gpu_data.host_fock[device_id] = zeros(Float64, (scf_data.μ, scf_data.μ))

            #cuda device data
            scf_data.gpu_data.device_fock[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64,(scf_data.μ, scf_data.μ))

            scf_data.gpu_data.device_density[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (p,p))
            scf_data.gpu_data.device_screened_density[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (scf_data.screening_data.screened_indices_count))
            scf_data.gpu_data.device_coulomb[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.device_coulomb_intermediate[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64,(Q))
        

            scf_data.gpu_data.device_occupied_orbital_coefficients[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (scf_data.occ, scf_data.μ))
            scf_data.gpu_data.device_non_zero_coefficients[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (n_ooc, p, p))
            scf_data.gpu_data.device_exchange_intermediate[device_id] =  GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (Q, n_ooc, p))
            lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_n_blocks)#should only be done on first iteration 
            scf_data.gpu_data.device_K_block[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (scf_data.screening_data.K_block_width, scf_data.screening_data.K_block_width, lower_triangle_length))
            
            ################   duplicated logic! move this to a shared place   ##########################
            row_nonsquare_range = p-(p%scf_options.df_exchange_n_blocks)+1:p
            scf_data.gpu_data.device_non_square_K_block[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (length(row_nonsquare_range), p))
            ############################################################################################################
            
            if rank == 0 && device_id == 1
                scf_data.gpu_data.device_H = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (scf_data.μ, scf_data.μ))
                copyto!(scf_data.gpu_data.device_H, H)
            end
            #timing gpu size in MB 
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
    
    fock_copy_time = 0.0


    n_threads = Threads.nthreads()
    threads_per_device = Int64((n_threads - num_devices) ÷ num_devices) 
    
    total_fock_gpu_time = @elapsed begin 
        Threads.@sync for device_id in 1:num_devices
            Threads.@spawn begin
                gpu_fock_times[device_id] = @elapsed begin 
                    set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)
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
                
                    copyto!(ooc, occupied_orbital_coefficients)
                    GPU_synchronize(scf_data.gpu_data.GPU_Type)

                    non_zero_coeff_times[device_id] = @elapsed form_nozero_coefficient_matrix!(scf_data, device_id)

                    # copy non_zero coefficients to the host and display
                    # if iteration == 1
                    #     println("non_zero_coefficients")
                    #     debug_non_zero_coeff = zeros(Float64, (n_ooc, p, p))
                    #     copyto!(debug_non_zero_coeff, scf_data.gpu_data.device_non_zero_coefficients[device_id])
                    #     GPU_synchronize(scf_data.gpu_data.GPU_Type)
                    #     display(debug_non_zero_coeff) 
                    # end


                    

                    W_times[device_id]  = @elapsed calculate_W_screened_GPU(device_id, scf_data, threads_per_device)
                    if scf_options.df_exchange_n_blocks > 1 
                        lower_triangle_length = get_triangle_matrix_length(scf_options.df_exchange_n_blocks)#should only be done on first iteration 
                        jc_timing.non_timing_data[JCTC.total_exchange_blocks] = string(lower_triangle_length)
                        K_times[device_id]  = @elapsed calculate_K_lower_diagonal_block_no_screen_GPU(host_fock, fock, W, Q_length, device_id,
                        scf_data, scf_options, lower_triangle_length, threads_per_device)       
                    else
                        K_times[device_id]  = @elapsed calcululate_K_no_sym_GPU!(fock, W, p, scf_data.occ, Q_length, device_id, scf_data.gpu_data.GPU_Type)
                    end

                    # #copy fock to the host and display
                    # if iteration == 1
                    #     println("exchange")
                    #     debug_fock = zeros(Float64, (scf_data.μ, scf_data.μ))
                    #     copyto!(debug_fock, fock)
                    #     GPU_synchronize(scf_data.gpu_data.GPU_Type)
                    #     display(debug_fock) 
                    # end
   
                    density_times[device_id]  = @elapsed form_screened_density!(scf_data, device_id)
                    V_times[device_id]  = @elapsed calculate_V_screened_GPU(V, B, density, scf_data.gpu_data.GPU_Type)
                    J_times[device_id]  = @elapsed calculate_J_screened_GPU(J, B, V, scf_data.gpu_data.GPU_Type)
                    
                    # J_times = @elapsed calculate_J_screened_symmetric_GPU(density, host_J, J, V, B, scf_data, device_id, n_j_streams_per_device)
                    # calculate_J_screened_symmetric_GPU(density, host_J, J, V, B, scf_data, device_id, n_j_streams_per_device)
                
                    # J_times = @elapsed calculate_J_screened_symmetric_GPU(density, host_J, J, V, B, scf_data, device_id, n_j_streams_per_device)
                    # GPU_synchronize(scf_data.gpu_data.GPU_Type)

                    gpu_copy_J_time[device_id] = @elapsed begin 
                        
                        device_sparse_to_p = scf_data.gpu_data.device_sparse_to_p[device_id]
                        device_sparse_to_q = scf_data.gpu_data.device_sparse_to_q[device_id]
    
                        # numblocks = ceil(Int64, scf_data.screening_data.screened_indices_count/256)
                        # threads = min(256, scf_data.screening_data.screened_indices_count)
                        # # @cuda threads=threads blocks=numblocks copy_screened_J_to_fock_upper_triangle(fock, J, device_sparse_to_p, 
                        #     device_sparse_to_q, scf_data.screening_data.screened_indices_count)
                        # amd @roc 
                        numThreads = 512
                        threads = min(scf_data.screening_data.screened_indices_count , numThreads)
                        blocks = ceil(Int, scf_data.screening_data.screened_indices_count / threads)

                        println("threads = ", threads, " blocks = ", blocks)

                        @roc groupsize=threads gridsize=blocks copy_screened_J_to_fock_upper_triangle_amd(fock, J, device_sparse_to_p, 
                            device_sparse_to_q, scf_data.screening_data.screened_indices_count)

                        GPU_synchronize(scf_data.gpu_data.GPU_Type) 

                    end
                    gpu_copy_sym_time[device_id] = @elapsed begin
    
                        device_sparse_to_p = scf_data.gpu_data.device_sparse_to_p[device_id]
                        device_sparse_to_q = scf_data.gpu_data.device_sparse_to_q[device_id]
    
                        # numblocks = ceil(Int64, scf_data.screening_data.screened_indices_count/256)
                        # threads = min(256, scf_data.screening_data.screened_indices_count)
                        # @cuda threads=threads blocks=numblocks copy_upper_to_lower_kernel(fock)

                        # amd @roc
                        numThreads = 512
                        threads = min(p , numThreads)
                        blocks = ceil(Int, p / threads)                        

                        @roc groupsize=threads gridsize=blocks copy_upper_to_lower_kernel_amd(fock)
                        GPU_synchronize(scf_data.gpu_data.GPU_Type) 
                    end
                    
                    # if iteration == 1
                    #     #copy fock and 
                    #     println("fock")
                    #     debug_fock = zeros(Float64, (scf_data.μ, scf_data.μ))
                    #     copyto!(debug_fock, fock)
                    #     display(debug_fock)
                    # end 

                    if rank == 0 && device_id == 1
                        H_add_time = @elapsed begin
                            axpy!(1.0, scf_data.gpu_data.device_H, fock)
                            GPU_synchronize(scf_data.gpu_data.GPU_Type)
                        end
                    end
                end # gpu fock time elapsed
            end #spawn     
        end#sync
    end# total fock gpu time elapsed

    fock_copy_time = @elapsed begin
        Threads.@threads for device_id in 1:num_devices
            set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)
            copyto!(scf_data.gpu_data.host_fock[device_id], scf_data.gpu_data.device_fock[device_id])  
            GPU_synchronize(scf_data.gpu_data.GPU_Type)
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
    jc_timing.timings[JCTiming_key(JCTC.fock_time, iteration)] = maximum(gpu_fock_times)
    jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_H_add_time, 1, iteration)] = H_add_time


    jc_timing.timings[JCTiming_key(JCTC.fock_gpu_cpu_copy_reduce_time, iteration)] = fock_copy_time
    jc_timing.timings[JCTiming_key(JCTC.total_fock_gpu_time, iteration)] = total_fock_gpu_time


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

#to remove branching I need a map from screened[1d index] to unscreened 2d[p,q] indices 
#not a huge performance hit at the moment so not proritiezed 
function form_screened_density_kernel_amd!(screened_density, density, sparse_pq_index_map, p::Int64)
    
    pp = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if pp > p
        return nothing
    end
    for qq in 1:pp-1
        if sparse_pq_index_map[pp, qq] == 0
            continue
        else 
            @inbounds screened_density[sparse_pq_index_map[pp, qq]] = 2.0*density[pp, qq] # symmetric multiplication 2.0* for off diagonal
        end
    end
    @inbounds screened_density[sparse_pq_index_map[pp, pp]] = density[pp, pp]  # and 1.0* for diagonal
    return nothing
end

function form_screened_density!(scf_data::SCFData, device_id::Int64)
    p = scf_data.μ
    density = scf_data.gpu_data.device_density[device_id]

    screened_density = scf_data.gpu_data.device_screened_density[device_id]
    occupied_orbital_coefficients = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]

    gemm!('T', 'N', 1.0, occupied_orbital_coefficients, occupied_orbital_coefficients, 0.0, density)
    GPU_synchronize(scf_data.gpu_data.GPU_Type)
    
    sparse_pq_index_map = scf_data.gpu_data.sparse_pq_index_map[device_id]

    # numblocks = ceil(Int64, p/256)
    # threads = min(256, p)

    # @cuda threads=threads blocks=numblocks form_screened_density_kernel!(screened_density, density, sparse_pq_index_map, p)
    
    #amd 
    numThreads = 512
    threads = min(p , numThreads)
    blocks = ceil(Int, p / threads)

    @roc groupsize=threads gridsize=blocks form_screened_density_kernel_amd!(screened_density, density, sparse_pq_index_map, p)
    GPU_synchronize(scf_data.gpu_data.GPU_Type)
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
            set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)
            scf_data.gpu_data.device_range_p[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Int, n_ranges)
            scf_data.gpu_data.device_range_start[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Int, n_ranges)
            scf_data.gpu_data.device_range_end[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Int, n_ranges)
            scf_data.gpu_data.device_range_sparse_start[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Int, n_ranges)
            scf_data.gpu_data.device_range_sparse_end[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Int, n_ranges)    
            scf_data.gpu_data.device_sparse_to_p[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Int64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.device_sparse_to_q[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Int64, scf_data.screening_data.screened_indices_count)
            
            scf_data.gpu_data.sparse_pq_index_map[device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Int, (p, p))

            copyto!(scf_data.gpu_data.sparse_pq_index_map[device_id], scf_data.screening_data.sparse_pq_index_map)
            copyto!(scf_data.gpu_data.device_range_p[device_id], range_p)
            copyto!(scf_data.gpu_data.device_range_start[device_id], range_start)
            copyto!(scf_data.gpu_data.device_range_end[device_id], range_end)
            copyto!(scf_data.gpu_data.device_range_sparse_start[device_id], range_sparse_start)
            copyto!(scf_data.gpu_data.device_range_sparse_end[device_id], range_sparse_end)
            GPU_synchronize(scf_data.gpu_data.GPU_Type) 
        end
    end

    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)

            # numblocks = ceil(Int64, n_ranges/256)
            # threads = min(256, n_ranges)
            # @cuda threads=threads blocks=numblocks create_sparse_to_p_q_kernel(scf_data.gpu_data.device_sparse_to_p[device_id],
            #     scf_data.gpu_data.device_sparse_to_q[device_id], 
            #     scf_data.gpu_data.sparse_pq_index_map[device_id], p)

            # inspired by https://github.com/JuliaORNL/JACC.jl/blob/main/ext/JACCAMDGPU/JACCAMDGPU.jl
            numThreads = 512
            threads = min(p , numThreads)
            blocks = ceil(Int, p / threads)

            @roc groupsize=threads gridsize=blocks create_sparse_to_p_q_kernel_amd(scf_data.gpu_data.device_sparse_to_p[device_id],
                scf_data.gpu_data.device_sparse_to_q[device_id], 
                scf_data.gpu_data.sparse_pq_index_map[device_id], p)

            GPU_synchronize(scf_data.gpu_data.GPU_Type)
        end 
    end
end
function create_sparse_to_p_q_kernel_cuda(sparse_to_p ::CuDeviceArray{Int64}, 
    sparse_to_q::CuDeviceArray{Int64}, 
    sparse_pq_index_map::CuDeviceArray{Int64},  
    p::Int64)
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

#todo make this a 2d kernel? 
function create_sparse_to_p_q_kernel_amd(sparse_to_p, 
    sparse_to_q,
    sparse_pq_index_map,  
    p::Int64)

    qq = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if qq > p
        return nothing
    end
    # stride = gridDim().x * blockDim().x

    for pp in 1:p
        @inbounds sparse_index = sparse_pq_index_map[pp, qq]
        if sparse_index != 0
            @inbounds sparse_to_p[sparse_index] = pp
            @inbounds sparse_to_q[sparse_index] = qq
        end
    end
end

function form_nozero_coefficient_matrix!(scf_data::SCFData, device_id :: Int64)
    set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)
    

    n_ranges = scf_data.gpu_data.n_screened_occupied_orbital_ranges

    # threads = min(256, n_ranges)
    # numblocks = ceil(Int64, n_ranges/256)

    # @cuda threads=threads blocks=numblocks build_non_zero_coefficients_kernel(scf_data.gpu_data.device_non_zero_coefficients[device_id], 
    #     scf_data.gpu_data.device_occupied_orbital_coefficients[device_id], 
    #     scf_data.gpu_data.device_range_p[device_id],
    #     scf_data.gpu_data.device_range_start[device_id],
    #     scf_data.gpu_data.device_range_end[device_id],
    #     scf_data.gpu_data.device_range_sparse_start[device_id],
    #     n_ranges)

    numThreads = 512
    threads = min(n_ranges , numThreads)
    blocks = ceil(Int64, n_ranges / threads)

    # println("threads = ", threads, " blocks = ", blocks)

    #copy occupied_orbital_coefficients to host and display

    # debug_occupied_orbital_coefficients = zeros(Float64, size(scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]))
    # copyto!(debug_occupied_orbital_coefficients, scf_data.gpu_data.device_occupied_orbital_coefficients[device_id])
    # GPU_synchronize(scf_data.gpu_data.GPU_Type)
    # println("occupied_orbital_coefficients from GPU")
    # display(debug_occupied_orbital_coefficients)


    @roc groupsize=threads gridsize=blocks build_non_zero_coefficients_kernel_amd(scf_data.gpu_data.device_non_zero_coefficients[device_id], 
        scf_data.gpu_data.device_occupied_orbital_coefficients[device_id], 
        scf_data.gpu_data.device_range_p[device_id],
        scf_data.gpu_data.device_range_start[device_id],
        scf_data.gpu_data.device_range_end[device_id],
        scf_data.gpu_data.device_range_sparse_start[device_id],
        n_ranges)

    GPU_synchronize(scf_data.gpu_data.GPU_Type)
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

function build_non_zero_coefficients_kernel_amd(non_zero_coefficients, 
     occupied_orbital_coefficients,
        device_range_p,
        device_range_start,
        device_range_end,
        device_range_sparse_start,
        n_ranges::Int64)

    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    
    if i > n_ranges
        return nothing
    end

    pp = device_range_p[i]
    range_start = device_range_start[i]
    range_end = device_range_end[i]
    range_sparse_start = device_range_sparse_start[i]
    # range_sparse_end = device_range_sparse_end[i]

    for j in 0:range_end-range_start
        for k in 1:size(non_zero_coefficients, 1)
            @inbounds non_zero_coefficients[k, range_sparse_start+j, pp] = occupied_orbital_coefficients[k, range_start+j]
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

function copy_screened_J_to_fock_upper_triangle_amd(fock, J,
    device_sparse_to_p, device_sparse_to_q, n_sparse_indicies::Int64)

    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x

    if i > n_sparse_indicies
        return nothing
    end

    qq = device_sparse_to_p[i]
    pp = device_sparse_to_q[i]
    @inbounds fock[pp, qq] += J[i]
    return nothing 
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

function copy_upper_to_lower_kernel_amd(A)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if i > size(A, 1)
        return nothing
    end
    for j = axes(A, 2)
        @inbounds A[j, i] = A[i, j] 
    end
    return nothing
end


function calculate_V_screened_GPU(V, B, density, gpu_type::GPU_Type)
    gemv!('N', 1.0, B, density, 0.0, V)
    GPU_synchronize(gpu_type)
end

function calculate_J_screened_GPU(J, B, V, gpu_type::GPU_Type)
    gemv!('T', 2.0, B, V, 0.0, J)
    GPU_synchronize(gpu_type)
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


    set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)
    for pp in 1:p
        K = scf_data.screening_data.non_screened_p_indices_count[pp]
        A_cu = view(B, :, scf_data.screening_data.sparse_p_start_indices[pp]:
            scf_data.screening_data.sparse_p_start_indices[pp]+K-1)
        B_cu = view(non_zero_coefficients, :,1:K,pp)
        C_cu = view(W, :,:,pp)
        gemm!('N','T', alpha, A_cu, B_cu, beta, C_cu)
    end

    GPU_synchronize(scf_data.gpu_data.GPU_Type)
end

function calcululate_K_no_sym_GPU!(fock, W, p::Int64, n_ooc::Int64, Q::Int64, device_id::Int64, gpu_type::GPU_Type)
    gemm!('T', 'N', -1.0, reshape(W, (Q*n_ooc, p)), reshape(W, (Q*n_ooc, p)), 0.0, fock)
    GPU_synchronize(gpu_type)
end

function calculate_K_lower_diagonal_block_no_screen_GPU(host_fock::Array{Float64,2},
     fock, W, Q_length::Int, 
     device_id, scf_data::SCFData, scf_options::SCFOptions,
     lower_triangle_length :: Int64, num_threads_avail::Int64)

    set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)

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

    set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)
    exchange_block = view(device_K_block, :,:, 1)

    for index in 1:lower_triangle_length
        pp, qq = scf_data.screening_data.exchange_batch_indexes[index]
        p_range = (pp-1)*K_block_width+1:pp*K_block_width        
        q_range = (qq-1)*K_block_width+1:qq*K_block_width

        A = reshape(view(W, :,:, p_range), (K, K_block_width))
        B = reshape(view(W, :,:, q_range), (K, K_block_width))

        gemm!(transA, transB, alpha, A, B, beta, exchange_block)
        GPU_synchronize(scf_data.gpu_data.GPU_Type)
        copyto!(view(fock, p_range, q_range), exchange_block)
        GPU_synchronize(scf_data.gpu_data.GPU_Type)

        #copy transpose 
        copyto!(view(fock, q_range, p_range), transpose(exchange_block))
        GPU_synchronize(scf_data.gpu_data.GPU_Type)
    end

    if p % scf_options.df_exchange_n_blocks != 0 # if square blocks don't cover the entire pq space
        col_non_square_range = 1:p    
        #non square part that didn't fit in blocks
        row_non_square_range = p-(p%scf_options.df_exchange_n_blocks)+1:p
        
        M = length(row_non_square_range)
        N = p
        
        A_non_square = reshape(view(W, :,:, row_non_square_range), (K, M))
        B_non_square = reshape(view(W, :,:, col_non_square_range), (K, N))
        C_non_square = scf_data.gpu_data.device_non_square_K_block[device_id]
        
    
        gemm!(transA, transB, alpha, A_non_square, B_non_square, beta, C_non_square) #W^T[M, Q*n_ooc] * W[Q*n_ooc, N] = C_non_square[M, N]
        GPU_synchronize(scf_data.gpu_data.GPU_Type)

        copyto!(view(fock, row_non_square_range,:), C_non_square)  #non contiguous memory access on the GPU bad, should use the other triangle side
        GPU_synchronize(scf_data.gpu_data.GPU_Type)

        #copy transpose
        copyto!(view(fock, :, row_non_square_range), transpose(C_non_square))
        GPU_synchronize(scf_data.gpu_data.GPU_Type)

    end 
end

function calculate_B_GPU!(two_center_integrals, three_center_integrals, scf_data, num_devices, num_devices_global, basis_sets, jc_timing, F_ARR_type :: Type)
    COMM = MPI.COMM_WORLD
    rank = MPI.Comm_rank(COMM)
    n_ranks = MPI.Comm_size(COMM)
    pq = scf_data.screening_data.screened_indices_count
    
    device_three_center_integrals = Array{F_ARR_type}(undef, num_devices)
    host_B_send_buffers = Array{Array{Float64}}(undef, num_devices)
    device_J_AB_invt = Array{F_ARR_type}(undef, num_devices)
    device_B_send_buffers = Array{F_ARR_type}(undef, num_devices)

    device_B = scf_data.gpu_data.device_B


    device_Q_indices, 
    device_rank_Q_indices, 
    device_Q_range_lengths, 
    max_device_Q_range_length  = calculate_device_ranges_GPU(scf_data, num_devices, n_ranks, basis_sets)

    scf_data.gpu_data.device_Q_range_lengths = device_Q_range_lengths
    scf_data.gpu_data.device_Q_indices = device_Q_indices

    device_id_offset = rank * num_devices
    
    if rank == 0

        set_gpu_device(0, scf_data.gpu_data.GPU_Type)
        device_J_AB_invt[1] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (scf_data.A, scf_data.A))

        J_AB_time = @elapsed begin
            copyto!(device_J_AB_invt[1], two_center_integrals)
            GPU_synchronize(scf_data.gpu_data.GPU_Type)
            potrf!('L', device_J_AB_invt[1])
            GPU_synchronize(scf_data.gpu_data.GPU_Type)
            GPU_trtri!(scf_data.gpu_data.GPU_Type, 'L', 'N', device_J_AB_invt[1])
            GPU_synchronize(scf_data.gpu_data.GPU_Type)
        end

        debug_jab = Array{Float64}(undef, size(device_J_AB_invt[1]))
        copyto!(debug_jab, device_J_AB_invt[1]) # copy back because taking subarrays on the GPU is slow / doesn't work. Need to look into if this is possible with CUDA.jl
        GPU_synchronize(scf_data.gpu_data.GPU_Type)

        jc_timing.timings[JCTC.form_J_AB_inv_time] = J_AB_time
    end

    # Threads.@sync for setup_device_id in 1:num_devices
    for setup_device_id in 1:num_devices
        # Threads.@spawn begin
            set_gpu_device(setup_device_id-1, scf_data.gpu_data.GPU_Type)
            global_device_id = setup_device_id  + device_id_offset
            # buffer for J_AB_invt for each device max size needed is A*A 
            # for certain B calculations the device will only need a subset of this
            # and will ref   device!(device_id - 1)reference it with a view referencing the front of the underlying array
            # device_J_AB_invt[setup_device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (scf_data.A, scf_data.A))

           
            #todo calculate the three center integrals per device (probably could directly copy to the device while it is being calculated)
            device_three_center_integrals[setup_device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type,Float64, size(three_center_integrals[setup_device_id]))
            copyto!(device_three_center_integrals[setup_device_id], three_center_integrals[setup_device_id])
            GPU_synchronize(scf_data.gpu_data.GPU_Type)

            device_B[setup_device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (device_Q_range_lengths[global_device_id], pq))
            GPU_synchronize(scf_data.gpu_data.GPU_Type)

            if num_devices_global > 1 
                device_B_send_buffers[setup_device_id] = GPU_zeros(scf_data.gpu_data.GPU_Type, Float64, (max_device_Q_range_length * pq))
                host_B_send_buffers[setup_device_id] = zeros(Float64, (max_device_Q_range_length * pq))
            end
            GPU_synchronize(scf_data.gpu_data.GPU_Type)
        # end #spawn
    end

   

    if MPI.Comm_size(COMM) > 1
        #broadcast two_center_integrals to all ranks
        MPI.Bcast!(two_center_integrals, 0, COMM)
    end
    

    if n_ranks == 1 && num_devices == 1
        set_gpu_device(0, scf_data.gpu_data.GPU_Type)

        B_time = @elapsed begin
            trmm!('L', 'L', 'N', 'N', 1.0, device_J_AB_invt[1], device_three_center_integrals[1], device_B[1])  
            GPU_synchronize(scf_data.gpu_data.GPU_Type) 
        end
        # CUDA.unsafe_free!(device_J_AB_invt[1])
        # CUDA.unsafe_free!(device_three_center_integrals[1])
        # CUDA.reclaim() todo put back reclaim 
        jc_timing.timings[JCTC.B_time] = B_time
        return
    end

    for device_id_two_eri in 2:num_devices
        set_gpu_device(device_id_two_eri-1, scf_data.gpu_data.GPU_Type)
        copyto!(device_J_AB_invt[device_id_two_eri], two_center_integrals)
        GPU_synchronize(scf_data.gpu_data.GPU_Type)
    end

    
    B_time = @elapsed begin
        for global_recieve_device_id in 1:num_devices_global
            rec_device_Q_range_length = device_Q_range_lengths[global_recieve_device_id]
            recieve_rank = (global_recieve_device_id-1) ÷ num_devices
            rank_recieve_device_id = ((global_recieve_device_id-1) % num_devices) + 1 # one indexed device id for the rank 
            array_size = rec_device_Q_range_length*pq
            Threads.@sync for r_send_device_id in 1:num_devices
                Threads.@spawn begin
                    set_gpu_device(r_send_device_id-1, scf_data.gpu_data.GPU_Type) 
                        global_send_device_id = r_send_device_id + device_id_offset 
                        send_device_Q_range_length = device_Q_range_lengths[global_send_device_id]
                        J_AB_invt_for_device = two_center_integrals[device_Q_indices[global_recieve_device_id],device_Q_indices[global_send_device_id]]
                        device_J_AB_inv_count = send_device_Q_range_length*rec_device_Q_range_length # total number of elements in the J_AB_invt matrix for the device
                        copyto!(device_J_AB_invt[r_send_device_id],1,J_AB_invt_for_device,1,device_J_AB_inv_count) #copy the needed J_AB_invt data to the device 

                        J_AB_INV_view = reshape(
                            view(device_J_AB_invt[r_send_device_id],
                            1:device_J_AB_inv_count), (rec_device_Q_range_length,send_device_Q_range_length))


                        if global_send_device_id == global_recieve_device_id
                            gemm!('N', 'N', 1.0, J_AB_INV_view, device_three_center_integrals[r_send_device_id], 1.0, device_B[rank_recieve_device_id])
                        else
                            send_B_view = view(device_B_send_buffers[r_send_device_id], 1:array_size)
                            gemm!('N', 'N', 1.0, J_AB_INV_view, device_three_center_integrals[r_send_device_id],
                                0.0, reshape(send_B_view, (rec_device_Q_range_length, pq)))
                        end
                        GPU_synchronize(scf_data.gpu_data.GPU_Type)
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
                    set_gpu_device(rank_send_device_id-1, scf_data.gpu_data.GPU_Type) 
                    copyto!(host_B_send_buffers[rank_send_device_id], 1, 
                    device_B_send_buffers[rank_send_device_id], 1, array_size)
                
                    

                    send_unique_tag = send_rank*100000 + recieve_rank*10000 + global_send_device_id*1000 + global_recieve_device_id*100
                    if send_rank == recieve_rank # devices belong to the same rank 
                        set_gpu_device(rank_recieve_device_id-1, scf_data.gpu_data.GPU_Type) 
                        copyto!(device_B_send_buffers[rank_recieve_device_id],
                            1, host_B_send_buffers[rank_send_device_id], 1, array_size)
                        
                    elseif rank == send_rank
                        set_gpu_device(rank_send_device_id-1, scf_data.gpu_data.GPU_Type) 
                        MPI.Send(host_B_send_buffers[rank_send_device_id], recieve_rank, send_unique_tag , COMM)
                    elseif rank == recieve_rank
                        set_gpu_device(rank_recieve_device_id-1, scf_data.gpu_data.GPU_Type)  
                        MPI.Recv!(host_B_send_buffers[rank_recieve_device_id], send_rank, send_unique_tag, COMM)
                        copyto!(device_B_send_buffers[rank_recieve_device_id],1,
                        host_B_send_buffers[rank_recieve_device_id],1, array_size)
                        GPU_synchronize(scf_data.gpu_data.GPU_Type)
                    end
                        
                    if rank == recieve_rank # add the sent buffer to the device B matrix
                        set_gpu_device(rank_recieve_device_id-1, scf_data.gpu_data.GPU_Type)  
                        device_B_send_view = reshape(view(device_B_send_buffers[rank_recieve_device_id], 1:array_size), (rec_device_Q_range_length, pq))
                        axpy!(1.0, device_B_send_view, device_B[rank_recieve_device_id])
                        GPU_synchronize(scf_data.gpu_data.GPU_Type)
                    end
                end
            end
        end
    end

    jc_timing.timings[JCTC.B_time] = B_time

    for device_id in 1:num_devices
        set_gpu_device(device_id-1, scf_data.gpu_data.GPU_Type)
        # CUDA.unsafe_free!(device_B_send_buffers[device_id])
        # CUDA.unsafe_free!(device_J_AB_invt[device_id])
        # CUDA.unsafe_free!(device_three_center_integrals[device_id]) # todo put back reclaim
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