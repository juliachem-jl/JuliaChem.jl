


#inherit from SCFGPUData
mutable struct SCFGPUData_generic{F_ARR, I_ARR} <: SCFGPUData
    device_Q_range_lengths::Array{Int,1}
    device_Q_range_starts::Array{Int,1}
    device_Q_range_ends::Array{Int,1}
    device_Q_indices::Array{UnitRange{Int},1}
    device_B::Array{F_ARR,1}
    device_B_send_buffers::Array{F_ARR,1}
    device_fock::Array{F_ARR,1}
    device_exchange_intermediate::Array{F_ARR,1}
    device_occupied_orbital_coefficients::Array{F_ARR,1}
    device_coulomb::Array{F_ARR,1}
    device_coulomb_intermediate::Array{F_ARR,1}
    device_density::Array{F_ARR,1}
    device_screened_density::Array{F_ARR,1}
    device_non_zero_coefficients::Array{F_ARR,1}
    device_K_block#::Array{F_ARR,1}
    device_non_square_K_block::Array{F_ARR,1}
    host_fock::Array{Array{Float64,2},1}
    device_H::F_ARR #only copied to rank 0 GPU 1 because it only needs to be added to one of the partial fock matricies 
    #metadata for screening 
    sparse_pq_index_map::Array{I_ARR,1}
    device_range_p::Array{I_ARR,1}
    device_range_start::Array{I_ARR,1}
    device_range_end::Array{I_ARR,1}
    device_range_sparse_start::Array{I_ARR,1}
    device_range_sparse_end::Array{I_ARR,1}
    device_sparse_to_p::Array{I_ARR,1}
    device_sparse_to_q::Array{I_ARR,1}
    n_screened_occupied_orbital_ranges::Int64
    number_of_devices_used::Int64
    device_Q_index_lengths::Array{Int,1}
    GPU_Type::GPU_Type
end


function initialize_generic!(F_ARR::Type, I_ARR::Type, gpu_data::SCFGPUData_generic, num_devices::Int64, gpu_type::GPU_Type)
    gpu_data.device_fock = Array{F_ARR}(undef, num_devices)
    gpu_data.device_coulomb_intermediate = Array{F_ARR}(undef, num_devices)
    gpu_data.device_coulomb = Array{F_ARR}(undef, num_devices)
    # gpu_data.device_stream_coulmob = Array{Array{F_ARR}}(undef, num_devices)

    gpu_data.device_B = Array{F_ARR}(undef, num_devices)

    gpu_data.device_exchange_intermediate = Array{F_ARR}(undef, num_devices)
    gpu_data.device_occupied_orbital_coefficients = Array{F_ARR}(undef, num_devices)
    gpu_data.device_density = Array{F_ARR}(undef, num_devices)
    gpu_data.device_screened_density = Array{F_ARR}(undef, num_devices)
    gpu_data.device_non_zero_coefficients = Array{Array{F_ARR}}(undef, num_devices)
    gpu_data.device_K_block = Array{F_ARR}(undef, num_devices)
    gpu_data.device_non_square_K_block = Array{F_ARR}(undef, num_devices)
    # gpu_data.host_coulomb = Array{Array{Float64,1}}(undef, num_devices)

    gpu_data.device_range_p = Array{I_ARR}(undef, num_devices)
    gpu_data.device_range_start = Array{I_ARR}(undef, num_devices)
    gpu_data.device_range_end = Array{I_ARR}(undef, num_devices)
    gpu_data.device_range_sparse_start = Array{I_ARR}(undef, num_devices)
    gpu_data.device_range_sparse_end = Array{I_ARR}(undef, num_devices)
    gpu_data.device_sparse_to_p = Array{I_ARR}(undef, num_devices)
    gpu_data.device_sparse_to_q = Array{I_ARR}(undef, num_devices)

    gpu_data.sparse_pq_index_map = Array{I_ARR}(undef, num_devices)

    gpu_data.host_fock = Array{Array{Float64,2}}(undef, num_devices)
    gpu_data.GPU_Type = gpu_type
end


export SCFGPUData_generic, initialize_generic!