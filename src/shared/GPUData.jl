
#parent struct for all GPUData structs

abstract type SCFGPUData end

abstract type GPU_Type end#abstract type for GPU_Type for aiding multiple dispatch overloading of GPU related functions 
struct GPU_Type_None <: GPU_Type end

mutable struct SCFGPUDataNoGPU <: SCFGPUData
end

function SCFGPUData()
end

function SCFGPUDataNone()
    return SCFGPUDataNoGPU()
end

export SCFGPUDataNone, SCFGPUDataNoGPU, GPU_Type_None