
#parent struct for all GPUData structs

abstract type SCFGPUData end

mutable struct SCFGPUDataNoGPU <: SCFGPUData
end

function SCFGPUData()
end

function SCFGPUDataNone()
    return SCFGPUDataNoGPU()
end

export SCFGPUDataNone, SCFGPUDataNoGPU