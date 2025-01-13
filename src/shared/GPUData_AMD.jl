using AMDGPU
using AMDGPU.rocBLAS
using LinearAlgebra

const RocAF64 = AMDGPU.ROCArray{Float64}
const RocAI64 = AMDGPU.ROCArray{Int64}
struct AMD_GPU <: GPU_Type end


function get_default_gpu_data_AMD() :: SCFGPUData_generic

    scf_data = SCFGPUData_generic{ROCArray{Float64}, ROCArray{Int64}}(
        [], [], [], [], [], 
        [], [], [], [], [], 
        [], [], [], [], [],
        [], [], [], [] ,[],
        [], [], [], [], [],
        [], 0, 0, [], AMD_GPU())
    initialize_generic!(RocAF64, RocAI64, scf_data, 1, AMD_GPU())
    return scf_data
end

function AMD_GPU_enabled()
    return AMDGPU.functional()
end

# set the device to the AMD GPU
# device_id is the device number in zero based indexign 
# GPU_Type is the type of GPU being used, parameter for aiding multiple dispatch
function set_gpu_device(device_id::Int64, gpu_type::AMD_GPU)
    AMDGPU.device_id!(device_id+1)
end

function get_array_types(gpu_type::AMD_GPU)
    return RocAF64, RocAI64
end

function GPU_zeros(gpu_type::AMD_GPU ,T::Type, dims::Any)
    return AMDGPU.zeros(T, dims...)
end

function GPU_synchronize(gpu_type::AMD_GPU)
    AMDGPU.synchronize()
end

function GPU_trtri!(gpu_type::AMD_GPU, uplo::Char, diag::Char, A::ROCArray{Float64})
    LinearAlgebra.LAPACK.chkuplo(uplo)
    n = LinearAlgebra.LAPACK.checksquare(A)
    lda = max(1, stride(A, 2))            
    devinfo = ROCVector{Cint}(undef, 1)

    AMDGPU.rocSOLVER.rocsolver_dtrtri(rocBLAS.handle(), uplo, diag, n, A, lda, devinfo)

    info = AMDGPU.@allowscalar devinfo[1]
    AMDGPU.unsafe_free!(devinfo)
    LinearAlgebra.LAPACK.chkargsok(LinearAlgebra.BlasInt(info))
end

function GPU_num_devices(gpu_type::AMD_GPU) :: Int64
    return length(AMDGPU.devices())
end


export get_default_gpu_data_AMD, AMD_GPU_enabled, set_gpu_device, get_array_types, GPU_zeros, GPU_synchronize, GPU_trtri!, GPU_num_devices