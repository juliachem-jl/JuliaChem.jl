using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using LinearAlgebra
using Base.Threads
using JuliaChem.Shared
using JuliaChem.Shared.JCTC

function df_rhf_fock_build_dense_GPU!(scf_data, jeri_engine_thread_df::Vector{T}, jeri_engine_thread::Vector{T2},
    basis_sets::CalculationBasisSets,
    occupied_orbital_coefficients, iteration, scf_options::SCFOptions, host_H::Array{Float64,2}, 
    jc_timing::JCTiming) where {T<:DFRHFTEIEngine,T2<:RHFTEIEngine}
    comm = MPI.COMM_WORLD
    pq = scf_data.μ^2
    rank = MPI.Comm_rank(comm)

    if rank == 0 && MPI.Comm_size(comm) > 1
        println("WARNING: Dense GPU algorithm only supports 1 rank runs, running on rank 0 only")        
    elseif rank != 0
        return
    end


    p = scf_data.μ
    n_ooc = scf_data.occ

    num_devices = scf_options.num_devices
    scf_data.gpu_data.number_of_devices_used = num_devices

    if iteration == 1
   

        Q_device_range_lengths = calculate_B_dense_GPU(scf_data, num_devices, jc_timing, jeri_engine_thread_df, jeri_engine_thread, basis_sets, scf_options)
        scf_data.gpu_data.device_Q_index_lengths = Q_device_range_lengths
        #clear the memory 

        scf_data.gpu_data.device_fock = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_coulomb_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_exchange_intermediate = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_occupied_orbital_coefficients = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.device_density = Array{CuArray{Float64}}(undef, num_devices)
        scf_data.gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)
        
        Threads.@threads for setup_device_id in 1:num_devices
            CUDA.device!(setup_device_id-1)
            Q = scf_data.gpu_data.device_Q_index_lengths[setup_device_id]

            scf_data.gpu_data.device_fock[setup_device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))
            scf_data.gpu_data.device_coulomb_intermediate[setup_device_id] = CUDA.zeros(Float64, (Q))
            scf_data.gpu_data.device_exchange_intermediate[setup_device_id] =
                CUDA.zeros(Float64, (n_ooc, Q, p))
            scf_data.gpu_data.device_occupied_orbital_coefficients[setup_device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.occ))
            scf_data.gpu_data.device_density[setup_device_id] = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))
            scf_data.gpu_data.host_fock[setup_device_id] = zeros(Float64, scf_data.μ, scf_data.μ)    
            if setup_device_id == 1
                scf_data.gpu_data.device_H = CUDA.zeros(Float64, (scf_data.μ, scf_data.μ))           
                CUDA.copyto!(scf_data.gpu_data.device_H, host_H)
            end
            CUDA.synchronize()   
            jc_timing.non_timing_data[JCTiming_GPUkey(JCTC.GPU_data_size_MB, setup_device_id)] = string(get_gpu_data_size_dense_MB(scf_data, setup_device_id))
       
        end

    

        jc_timing.non_timing_data[JCTC.contraction_algorithm] = "dense gpu"
        jc_timing.non_timing_data[JCTC.GPU_num_devices] = string(num_devices)
    end



    gpu_fock_times = zeros(Float64, num_devices)
    density_times = zeros(Float64, num_devices)
    GPU_H_add_time = 0.0
    timings_dicts = Vector{Dict{String, Float64}}(undef, num_devices)
    total_fock_gpu_time = @elapsed begin
        if num_devices > 1 #performance tweak for small systems and single device don't use threads because it intermittenly takes longer to spin up threads than to build the fock matrix
            Threads.@threads for device_id in 1:num_devices 
                timings_dicts[device_id] = fock_build_kernel_dense_GPU(device_id, scf_data, occupied_orbital_coefficients, iteration, scf_options, jc_timing)
            end
        else 
            timings_dicts[1] = fock_build_kernel_dense_GPU(1, scf_data, occupied_orbital_coefficients, iteration, scf_options, jc_timing)
        end
    end # end total_fock_gpu_time

    for timing_dict in timings_dicts
        for timing_key in keys(timing_dict)
            jc_timing.timings[timing_key] = timing_dict[timing_key]
        end
    end

    fock_copy_time = @elapsed begin
        if num_devices > 1
            Threads.@threads for device_id in 1:num_devices
                CUDA.copyto!(scf_data.gpu_data.host_fock[device_id], scf_data.gpu_data.device_fock[device_id])
            end
        else #performance tweak for small systems and single device don't use threads because it intermittenly takes longer to spin up threads than to build the fock matrix
            CUDA.copyto!(scf_data.gpu_data.host_fock[1], scf_data.gpu_data.device_fock[1])
        end
        scf_data.two_electron_fock = scf_data.gpu_data.host_fock[1]
        for device_id in 2:num_devices
            axpy!(1.0, scf_data.gpu_data.host_fock[device_id], scf_data.two_electron_fock)
        end
    end

    # jc_timing.timings[JCTiming_key(JCTC.K_time, iteration)] = maximum(K_times)
    # jc_timing.timings[JCTiming_key(JCTC.W_time, iteration)] = maximum(W_times)
    # jc_timing.timings[JCTiming_key(JCTC.V_time, iteration)] = maximum(V_times)
    # jc_timing.timings[JCTiming_key(JCTC.J_time, iteration)] = maximum(J_times)
    jc_timing.timings[JCTiming_key(JCTC.fock_time, iteration)] = total_fock_gpu_time + fock_copy_time
    jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_H_add_time, 1, iteration)] = GPU_H_add_time


    jc_timing.timings[JCTiming_key(JCTC.fock_gpu_cpu_copy_reduce_time, iteration)] = fock_copy_time
    jc_timing.timings[JCTiming_key(JCTC.total_fock_gpu_time, iteration)] = total_fock_gpu_time

end

function fock_build_kernel_dense_GPU(device_id, scf_data, occupied_orbital_coefficients, iteration, scf_options, jc_timing)
    CUDA.device!(device_id-1)
    Q = scf_data.gpu_data.device_Q_index_lengths[device_id]
    n_ooc = scf_data.occ 
    ooc = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
    density = scf_data.gpu_data.device_density[device_id]
    pq = scf_data.μ^2
    p = scf_data.μ


    B = scf_data.gpu_data.device_B[device_id]
    V = scf_data.gpu_data.device_coulomb_intermediate[device_id]
    W = scf_data.gpu_data.device_exchange_intermediate[device_id]
    fock = scf_data.gpu_data.device_fock[device_id]
    
    density_time = 0.0
    V_time = 0.0
    W_time = 0.0
    J_time = 0.0
    K_time = 0.0
    GPU_H_add_time = 0.0
    gpu_fock_time = @elapsed begin
        CUDA.copyto!(ooc, occupied_orbital_coefficients)

        density_time = @elapsed begin 
            CUDA.CUBLAS.gemm!('N', 'T', 1.0, ooc, ooc, 0.0, density)
            CUDA.synchronize()   

        end
        V_time = @elapsed begin
            CUDA.CUBLAS.gemv!('N', 1.0, reshape(B, (Q, pq)), reshape(density, pq), 0.0, V)
            CUDA.synchronize()   
        end    
        J_time = @elapsed begin
            CUDA.CUBLAS.gemv!('T', 2.0, reshape(B, (Q, pq)), V, 0.0, reshape(fock, pq))
            CUDA.synchronize()   
        end
        W_time = @elapsed begin
            CUDA.CUBLAS.gemm!('T', 'T', 1.0, ooc, reshape(B, (Q * p, p)), 0.0, reshape(W, (n_ooc, Q* p)))
            CUDA.synchronize()   
        end
        K_time = @elapsed begin
            CUDA.CUBLAS.gemm!('T', 'N', -1.0, reshape(W, (n_ooc * Q, p)), reshape(W, (n_ooc * Q, p)), 1.0, fock)
            CUDA.synchronize()   
        end
        if device_id == 1
            GPU_H_add_time = @elapsed begin
                CUDA.axpy!(1.0, scf_data.gpu_data.device_H, fock)
                CUDA.synchronize()   
            end
        end
    end # end gpu_fock_time

    timings = Dict{String, Float64}()

    timings[JCTiming_GPUkey(JCTC.GPU_W_time, device_id, iteration)] = W_time
    timings[JCTiming_GPUkey(JCTC.GPU_V_time, device_id, iteration)] = V_time
    timings[JCTiming_GPUkey(JCTC.GPU_J_time, device_id, iteration)] = J_time
    timings[JCTiming_GPUkey(JCTC.GPU_K_time, device_id, iteration)] = K_time
    timings[JCTiming_GPUkey(JCTC.GPU_H_add_time, device_id, iteration)] = GPU_H_add_time
    timings[JCTiming_GPUkey(JCTC.GPU_density_time, device_id, iteration)] = density_time
    timings[JCTiming_GPUkey(JCTC.gpu_fock_time, device_id, iteration)] = gpu_fock_time
    return timings
end

function copy_sparse_to_dense_B_kernel!(dense_B::CuDeviceArray{Float64}, sparse_B::CuDeviceArray{Float64},
    sparse_to_p::CuDeviceArray{Int64}, sparse_to_q::CuDeviceArray{Int64},
    screened_count::Int64, device_num_Q::Int64)

    pq_prime = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    aux = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if pq_prime <= screened_count && aux <= device_num_Q
        pp = sparse_to_p[pq_prime]
        qq = sparse_to_q[pq_prime]
        @inbounds dense_B[aux, pp, qq] = sparse_B[aux, pq_prime]
    end
    return
end

function copy_sparse_to_dense_B!(scf_data, num_devices)
    screened_count = scf_data.screening_data.screened_indices_count
    μ = scf_data.μ
    p = scf_data.μ
    scf_data.gpu_data.device_sparse_to_p = Array{CuArray{Int64,1}}(undef, num_devices)
    scf_data.gpu_data.device_sparse_to_q = Array{CuArray{Int64,1}}(undef, num_devices)
    scf_data.gpu_data.sparse_pq_index_map = Array{CuArray{Int,2}}(undef, num_devices)

    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            scf_data.gpu_data.device_sparse_to_p[device_id] = CUDA.zeros(Int64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.device_sparse_to_q[device_id] = CUDA.zeros(Int64, scf_data.screening_data.screened_indices_count)
            scf_data.gpu_data.sparse_pq_index_map[device_id] = CUDA.zeros(Int, (p, p))
            copyto!(scf_data.gpu_data.sparse_pq_index_map[device_id], scf_data.screening_data.sparse_pq_index_map)
        end
    end
    CUDA.synchronize()

    run_create_sparse_to_p_q_kernel(scf_data, num_devices, p, p)
    
    Threads.@sync for device_id in 1:num_devices
        Threads.@spawn begin
            device_num_Q = scf_data.gpu_data.device_Q_range_lengths[device_id] # Assuming all devices have the same Q range length

            d_sparse_to_p = scf_data.gpu_data.device_sparse_to_p[device_id]
            d_sparse_to_q = scf_data.gpu_data.device_sparse_to_q[device_id]

            sparse_B = scf_data.gpu_data.device_B[device_id]
            dense_B  = CUDA.zeros(Float64, (device_num_Q, μ, μ))

            threads_x = 32
            threads_y = 8
            blocks_x  = ceil(Int64, screened_count / threads_x)
            blocks_y  = ceil(Int64, device_num_Q / threads_y)

            @cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) copy_sparse_to_dense_B_kernel!(
                dense_B, sparse_B, d_sparse_to_p, d_sparse_to_q, screened_count, device_num_Q)
            
            scf_data.gpu_data.device_B[device_id] = dense_B
            
        end
    end
    CUDA.synchronize()

end

function calculate_B_dense_GPU(scf_data, num_devices, jc_timing::JCTiming, jeri_engine_thread_df, jeri_engine_thread,basis_sets, scf_options)

    n_ranks = MPI.Comm_size(MPI.COMM_WORLD)
    two_eri_time = @elapsed two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
    jc_timing.timings[JCTiming_key(JCTC.two_eri_time, 1)] = two_eri_time

    use_screening = true
    if scf_options.df_screening_sigma != 0.0
        get_screening_metadata!(scf_data, scf_options.df_screening_sigma, 
                jeri_engine_thread, two_center_integrals, basis_sets, jc_timing)
    else
        setup_unscreened_screening_matricies(basis_sets, scf_data)
    end

    scf_data.gpu_data.device_B = Array{CuArray{Float64}}(undef, num_devices)
    scf_data.gpu_data.device_B_send_buffers = Array{CuArray{Float64}}(undef, num_devices)
    pq = scf_data.μ^2

    device_Q_index_lengths = zeros(Int, num_devices)
    aux_ranges = Array{UnitRange{Int}}(undef, num_devices)
    Threads.@threads for device_id in 1:num_devices
        device_shell_aux_indicies, 
        device_aux_indicies, 
        device_basis_index_map = static_load_rank_indicies(device_id-1,num_devices,basis_sets) 
        aux_ranges[device_id] = device_aux_indicies
        device_Q_index_lengths[device_id] = length(device_aux_indicies)
    end
    num_devices_global = num_devices*n_ranks  
    use_screening = true

    device_Q_indices, 
    device_rank_Q_indices, 
    device_Q_range_lengths, 
    max_device_Q_range_length  = calculate_device_ranges_GPU(scf_data, num_devices, n_ranks, basis_sets)

    scf_data.gpu_data.device_Q_range_lengths = device_Q_range_lengths
    scf_data.gpu_data.device_Q_indices = device_Q_indices
    calculate_B_GPU_Screened!(two_center_integrals, 
        scf_data, 
        num_devices, 
        num_devices_global, 
        max_device_Q_range_length,
        jc_timing, jeri_engine_thread_df, 
        basis_sets, scf_options)
    if scf_data.screening_data.screened_indices_count != pq
        copy_sparse_to_dense_B!(scf_data, num_devices)
    end
    return device_Q_range_lengths

end

function calculate_device_ranges_dense(scf_data, num_devices)
    indices_per_device = scf_data.A ÷ num_devices
    device_Q_range_starts = []
    device_Q_range_ends = []
    device_Q_range_lengths = []

    for device_id in 1:num_devices
        push!(device_Q_range_starts, (device_id - 1) * indices_per_device + 1)
        push!(device_Q_range_ends, device_id * indices_per_device)
    end

    device_Q_range_ends[end] = scf_data.A

    # device_Q_range_starts = 1:indices_per_device+1:scf_data.A
    # device_Q_range_ends = device_Q_range_starts .+ indices_per_device
    
    device_Q_indices = [device_Q_range_starts[i]:device_Q_range_ends[i] for i in 1:num_devices]
    device_Q_indices[end] = device_Q_range_starts[end]:scf_data.A
    
    device_Q_range_lengths = length.(device_Q_indices)
    
    max_device_Q_range_length = maximum(device_Q_range_lengths)
    return indices_per_device, device_Q_range_starts, device_Q_range_ends, device_Q_indices, device_Q_range_lengths, max_device_Q_range_length
end

function get_gpu_data_size_dense_MB(scf_data::SCFData, device_id)
    gpu_data_size_MB = 0.0

    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_B[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_coulomb_intermediate[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_exchange_intermediate[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_occupied_orbital_coefficients[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_density[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_fock[device_id])
    gpu_data_size_MB += sizeof(scf_data.gpu_data.device_H)

    
    return gpu_data_size_MB / 1024^2
end