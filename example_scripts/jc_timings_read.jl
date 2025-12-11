using HDF5
include("../src/shared/JCTiming.jl")

function jc_timings_read(path::String)
    run_level_data = Dict()
    timings = Dict()
    scf_options = Dict()
    scf_options_user = Dict()
    non_timing_data = Dict()

    # read in the data
    h5open(path, "r") do file
        run_level_data = read(file, JCTC.run_level_data)
        timings = read(file, JCTC.timings)
        scf_options = read(file, JCTC.scf_options)
        scf_options_user = read(file, JCTC.scf_options_user)
        non_timing_data = read(file, JCTC.non_timing_data)
    end

    return run_level_data, timings, scf_options, scf_options_user, non_timing_data
end

