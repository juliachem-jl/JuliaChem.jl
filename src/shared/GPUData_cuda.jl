using CUDA

struct CUDA_GPU <: GPU_Type end


#inherit from SCFGPUData
mutable struct SCFGPUData_cuda <: SCFGPUData
    device_Q_range_lengths::Array{Int,1}
    device_Q_range_starts::Array{Int,1}
    device_Q_range_ends::Array{Int,1}
    device_Q_indices::Array{UnitRange{Int},1}
    device_B::Array{Union{Nothing,CuArray{Float64}},1}
    device_B_send_buffers::Array{Union{Nothing,CuArray{Float64}},1}
    device_fock::Array{Union{Nothing,CuArray{Float64}},1}
    device_exchange_intermediate::Array{Union{Nothing,CuArray{Float64}},1}
    device_occupied_orbital_coefficients::Array{Union{Nothing,CuArray{Float64}},1}
    device_coulomb::Array{Union{Nothing,CuArray{Float64}},1}
    device_coulomb_intermediate::Array{Union{Nothing,CuArray{Float64}},1}
    device_density::Array{Union{Nothing,CuArray{Float64}},1}
    device_screened_density::Array{Union{Nothing,CuArray{Float64}},1}
    device_non_zero_coefficients::Array{Union{Nothing,CuArray{Float64}},1}
    device_K_block#::Array{Union{Nothing,CuArray{Float64}},1}
    device_non_square_K_block::Array{Union{Nothing,CuArray{Float64}},1}
    host_fock::Array{Array{Float64,2},1}
    device_H::CuArray{Float64} #only copied to rank 0 GPU 1 because it only needs to be added to one of the partial fock matricies 
    #metadata for screening 
    sparse_pq_index_map::Array{CuArray{Int64,2},1}
    device_range_p::Array{CuArray{Int64,1},1}
    device_range_start::Array{CuArray{Int64,1},1}
    device_range_end::Array{CuArray{Int64,1},1}
    device_range_sparse_start::Array{CuArray{Int64,1},1}
    device_range_sparse_end::Array{CuArray{Int64,1},1}
    device_sparse_to_p::Array{CuArray{Int64,1},1}
    device_sparse_to_q::Array{CuArray{Int64,1},1}
    n_screened_occupied_orbital_ranges::Int64
    number_of_devices_used::Int64
    device_Q_index_lengths::Array{Int,1}
end

function get_default_gpu_data_cuda() :: SCFGPUData_cuda
    return SCFGPUData_cuda([], [], [], [], [], [], [], [], [],
        [], [], [], [], [], [], [], [], [],
        [], [], [], [], [], [],
        CuArray{Float64}(undef, 0), [], 0, 0, [])

end

function CUDA_GPU_enabled()
    return CUDA.functional()
end

function set_gpu_device(device_id::Int64, gpu_type::CUDA_GPU)
    CUDA.device!(device_id)
end

function initialize!(gpu_data::SCFGPUData_cuda, num_devices::Int64)
    gpu_data.device_fock = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_coulomb_intermediate = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_coulomb = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_stream_coulmob = Array{Array{CuArray{Float64}}}(undef, num_devices)


    gpu_data.device_exchange_intermediate = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_occupied_orbital_coefficients = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_density = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_screened_density = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_non_zero_coefficients = Array{Array{CuArray{Float64}}}(undef, num_devices)
    gpu_data.device_K_block = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.device_non_square_K_block = Array{CuArray{Float64}}(undef, num_devices)
    gpu_data.host_coulomb = Array{Array{Float64,1}}(undef, num_devices)

    gpu_data.device_range_p = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_range_start = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_range_end = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_range_sparse_start = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_range_sparse_end = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_sparse_to_p = Array{CuArray{Int64,1}}(undef, num_devices)
    gpu_data.device_sparse_to_q = Array{CuArray{Int64,1}}(undef, num_devices)

    gpu_data.sparse_pq_index_map = Array{CuArray{Int64,2}}(undef, num_devices)

    gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)

end

function GPU_trtri!(gpu_type::CUDA_GPU, uplo::Char, diag::Char, A::CuArray{Float64})

    CUBLAS.trtri!(uplo, diag, A)

end

export initialize!, get_default_gpu_data_cuda, SCFGPUData_cuda, CUDA_GPU_enabled, set_gpu_device, GPU_trtri!