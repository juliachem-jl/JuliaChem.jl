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
   

        Q_device_range_lengths = calculate_B_dense_GPU(scf_data, num_devices, jc_timing, jeri_engine_thread_df, basis_sets, scf_options)
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


    J_times = zeros(Float64, num_devices)
    W_times = zeros(Float64, num_devices)
    K_times = zeros(Float64, num_devices)
    V_times = zeros(Float64, num_devices)
    gpu_fock_times = zeros(Float64, num_devices)
    density_times = zeros(Float64, num_devices)
    GPU_H_add_time = 0.0
    total_fock_gpu_time = @elapsed begin
        Threads.@threads for device_id in 1:num_devices
            CUDA.device!(device_id-1)
            Q = scf_data.gpu_data.device_Q_index_lengths[device_id]
            ooc = scf_data.gpu_data.device_occupied_orbital_coefficients[device_id]
            density = scf_data.gpu_data.device_density[device_id]
            
            B = scf_data.gpu_data.device_B[device_id]
            V = scf_data.gpu_data.device_coulomb_intermediate[device_id]
            W = scf_data.gpu_data.device_exchange_intermediate[device_id]
            fock = scf_data.gpu_data.device_fock[device_id]
            
            gpu_fock_times[device_id] = @elapsed begin
                CUDA.copyto!(ooc, occupied_orbital_coefficients)
                CUDA.synchronize()

                density_times[device_id] = @elapsed begin 
                    CUDA.CUBLAS.gemm!('N', 'T', 1.0, ooc, ooc, 0.0, density)
                    CUDA.synchronize()          
                end
                V_times[device_id] = @elapsed begin
                    CUDA.CUBLAS.gemv!('N', 1.0, reshape(B, (Q, pq)), reshape(density, pq), 0.0, V)
                    CUDA.synchronize()      
                end    
                J_times[device_id] = @elapsed begin
                    CUDA.CUBLAS.gemv!('T', 2.0, reshape(B, (Q, pq)), V, 0.0, reshape(fock, pq))
                    CUDA.synchronize()          
                end
                W_times[device_id] = @elapsed begin
                    CUDA.CUBLAS.gemm!('T', 'T', 1.0, ooc, reshape(B, (Q * p, p)), 0.0, reshape(W, (n_ooc, Q* p)))
                    CUDA.synchronize()  
                end
                K_times[device_id] = @elapsed begin
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
        end # end Threads.@threads
        
     
    end # end total_fock_gpu_time


    fock_copy_time = @elapsed begin
        CUDA.copyto!(scf_data.gpu_data.host_fock[1], scf_data.gpu_data.device_fock[1])
        scf_data.two_electron_fock = scf_data.gpu_data.host_fock[1]
        Threads.@threads for device_id in 2:num_devices
            CUDA.copyto!(scf_data.gpu_data.host_fock[device_id], scf_data.gpu_data.device_fock[device_id])
        end
        for device_id in 2:num_devices
            axpy!(1.0, scf_data.gpu_data.host_fock[device_id], scf_data.two_electron_fock)
        end
    end

 


    for device_id in 1:num_devices
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_W_time, device_id, iteration)] = W_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_V_time, device_id, iteration)] = V_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_J_time, device_id, iteration)] = J_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_K_time, device_id, iteration)] = K_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_density_time, device_id, iteration)] = density_times[device_id]
        jc_timing.timings[JCTiming_GPUkey(JCTC.gpu_fock_time, device_id, iteration)] = gpu_fock_times[device_id]
    end

    jc_timing.timings[JCTiming_key(JCTC.K_time, iteration)] = maximum(K_times)
    jc_timing.timings[JCTiming_key(JCTC.W_time, iteration)] = maximum(W_times)
    jc_timing.timings[JCTiming_key(JCTC.V_time, iteration)] = maximum(V_times)
    jc_timing.timings[JCTiming_key(JCTC.J_time, iteration)] = maximum(J_times)
    jc_timing.timings[JCTiming_key(JCTC.fock_time, iteration)] = maximum(gpu_fock_times)
    jc_timing.timings[JCTiming_GPUkey(JCTC.GPU_H_add_time, 1, iteration)] = GPU_H_add_time


    jc_timing.timings[JCTiming_key(JCTC.fock_gpu_cpu_copy_reduce_time, iteration)] = fock_copy_time
    jc_timing.timings[JCTiming_key(JCTC.total_fock_gpu_time, iteration)] = total_fock_gpu_time

end

function calculate_B_dense_GPU(scf_data, num_devices, jc_timing::JCTiming, jeri_engine_thread_df, basis_sets, scf_options)

    two_eri_time = @elapsed two_center_integrals = calculate_two_center_intgrals(jeri_engine_thread_df, basis_sets, scf_options)
    jc_timing.timings[JCTiming_key(JCTC.two_eri_time, 1)] = two_eri_time


    scf_data.gpu_data.device_B = Array{CuArray{Float64}}(undef, num_devices)
    device_B = scf_data.gpu_data.device_B
    device_three_center_integrals = Array{CuArray{Float64}}(undef, num_devices)
    scf_data.gpu_data.device_B_send_buffers = Array{CuArray{Float64}}(undef, num_devices)
    device_J_AB_invt = Array{CuArray{Float64}}(undef, num_devices)
    
    #always do J_AB_INV on the first device
    for device_id in 1:num_devices
        CUDA.device!(device_id-1)
        device_J_AB_invt[device_id] = CUDA.zeros(Float64, (scf_data.A, scf_data.A))
        CUDA.synchronize()
    end
    
   
    form_J_AB_inv_time = @elapsed begin
        CUDA.copyto!(device_J_AB_invt[1], two_center_integrals)
        CUDA.synchronize()
        CUDA.CUSOLVER.potrf!('L', device_J_AB_invt[1])
        CUDA.synchronize()
        CUDA.CUSOLVER.trtri!('L', 'N',  device_J_AB_invt[1])
        CUDA.synchronize()        
    end
    jc_timing.timings[JCTC.form_J_AB_inv_time] = form_J_AB_inv_time

    pq = scf_data.μ^2
    if num_devices == 1
        three_eri_time = @elapsed other_device_three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options, 
        scf_data, 0, 1, false, false)
        device_three_center_integrals[1] = CUDA.zeros(Float64, (scf_data.A, pq))
        device_B[1] = CUDA.zeros(Float64, (scf_data.A, pq))
        CUDA.synchronize()
        CUDA.copyto!(device_three_center_integrals[1], other_device_three_center_integrals)
        CUDA.synchronize()
        
        B_time = @elapsed begin
            
            CUDA.CUBLAS.trmm!('L', 'L', 'N', 'N', 1.0, device_J_AB_invt[1], device_three_center_integrals[1], device_B[1])   
            CUDA.synchronize()
        end
        jc_timing.timings[JCTiming_key(JCTC.three_eri_time, 1)] = three_eri_time
        jc_timing.timings[JCTC.B_time] = B_time

        CUDA.unsafe_free!(device_J_AB_invt[1])
        CUDA.unsafe_free!(device_three_center_integrals[1])
        CUDA.reclaim()

        return [scf_data.A]
    else
        setup_unscreened_screening_matricies(basis_sets, scf_data)
        #copy J_AB_INV to all devices
        CUDA.copyto!(two_center_integrals, device_J_AB_invt[1])
        Threads.@threads for device_id in 2:num_devices
            CUDA.device!(device_id-1)
            CUDA.copyto!(device_J_AB_invt[device_id], two_center_integrals)
            CUDA.synchronize()
        end

        device_Q_index_lengths = zeros(Int, num_devices)
        aux_ranges = Array{UnitRange{Int}}(undef, num_devices)
        Threads.@threads for device_id in 1:num_devices
            device_shell_aux_indicies, 
            device_aux_indicies, 
            device_basis_index_map = static_load_rank_indicies(device_id-1,num_devices,basis_sets) 
            aux_ranges[device_id] = device_aux_indicies
            device_Q_index_lengths[device_id] = length(device_aux_indicies)
        end

        max_device_Q_range_length = maximum(device_Q_index_lengths)

        Threads.@threads for device_id in 1:num_devices
            CUDA.device!(device_id-1)
            device_three_center_integrals[device_id] = CUDA.zeros(Float64, (max_device_Q_range_length*pq))
            device_B[device_id] = CUDA.zeros(Float64, (device_Q_index_lengths[device_id],pq))
        end


        
        for other_device_id in 1:num_devices
            other_device_aux_indicies = aux_ranges[other_device_id]
      
            three_eri_time = @elapsed begin
              other_device_three_center_integrals = calculate_three_center_integrals(jeri_engine_thread_df, basis_sets, scf_options,
                scf_data, other_device_id-1, num_devices, true, false)
            end

            Threads.@threads for device_id in 1:num_devices
                CUDA.device!(device_id-1)
                device_aux_indicies = aux_ranges[device_id]
                three_eri_view = view(device_three_center_integrals[device_id], 1:(device_Q_index_lengths[other_device_id]*pq))
                reshape_three_eri_view = reshape(three_eri_view, (device_Q_index_lengths[other_device_id], pq))
                CUDA.copyto!(reshape_three_eri_view, other_device_three_center_integrals)
                CUDA.synchronize()

                rank_rank_J_AB_invt = CUDA.zeros(Float64, (device_Q_index_lengths[device_id], device_Q_index_lengths[other_device_id]))
                CUDA.synchronize()
                CUDA.copyto!(rank_rank_J_AB_invt, view(device_J_AB_invt[device_id], device_aux_indicies, other_device_aux_indicies))
                CUDA.synchronize()
                CUDA.CUBLAS.gemm!('N', 'N', 1.0, rank_rank_J_AB_invt, reshape_three_eri_view, 
                    1.0, device_B[device_id])
                CUDA.synchronize()  
            end
        end      


        return device_Q_index_lengths
    end

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