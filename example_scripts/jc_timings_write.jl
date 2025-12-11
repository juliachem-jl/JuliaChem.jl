using HDF5
include("../src/shared/JCTiming.jl")

#docs 
# Take the data from JCtiming and save it to an hdf5 file which does not need 
# The rest of JuliaChem to be read, only the JCTC keys 
function save_jc_timings_to_hdf5(jc_timing, file_path::String)
    # write stuff struct to hdf5 
    h5open("$(file_path)", "w") do file
        # write_dictionary_to_hdf5{String}(file, jc_timing.non_timing_data, "non_timing_data")
        save_scf_options(file, jc_timing.options, JCTC.scf_options) # scf options
        save_scf_options(file, jc_timing.user_options, "$(JCTC.scf_options)-user") # scf options
        save_run_level_data(file, jc_timing) #run level data
        save_timings(file, jc_timing)
        save_non_timing_data(file, jc_timing)

    end
end

function save_non_timing_data(file, jc_timing)
    # write the non timing data to the hdf5 file
    str_array = Array{String}(undef, length(jc_timing.non_timing_data), 2)
    row = 1
    for (key, value) in sort(collect(pairs(jc_timing.non_timing_data)), by= x->x[1])
        str_array[row, 1] = key
        str_array[row, 2] = value
        row += 1
    end
    write(file, JCTC.non_timing_data, str_array)

end

function save_timings(file, jc_timing)
    # write the timings to the hdf5 file

    #sort the items in the dictionary jc_timing.timings by key then iteration 
    sorted_timings = sort_timings(jc_timing)
    write(file, JCTC.timings, sorted_timings)
end

function sort_timings(jc_timing)
    #sort the items in the dictionary jc_timing.timings by key then iteration 

    timings_iteration_dict = Dict{Int, Array{Any}}()
    non_iteration_dict = Dict{String, Float64}()
    for (key, value) in jc_timing.timings
        key_split = split(key, "-")
        if length(key_split) >= 2
            iteration_str = key_split[end]
            iteration = parse(Int, iteration_str)
            if haskey(timings_iteration_dict, iteration)
                push!(timings_iteration_dict[iteration], (key, value))
            else
                timings_iteration_dict[iteration] = [(key, value)]
            end
        else
            #put in separate non iteration dictionary
            non_iteration_dict[key] = value
        end
    end

    sorted_timings = Array{String}(undef, length(jc_timing.timings), 2)

    row = 1
    for (iter_key, value) in sort(collect(pairs(timings_iteration_dict)), by= x->x[1])
        for (key, value) in sort(value, by= x->x[1])
            sorted_timings[row, 1] = string(key)
            sorted_timings[row, 2] = string(value)
            row += 1
        end
    end
    for (key, value) in sort(collect(pairs(non_iteration_dict)), by= x->x[1])
        sorted_timings[row, 1] = string(key)
        sorted_timings[row, 2] = string(value)
        row += 1
    end


    return sorted_timings
end

function save_scf_options(file, options, file_name)
    # write the scf options to the hdf5 file
    str_array = Array{String}(undef, 18, 2)

    str_array[1, 1] = JCTC.density_fitting
    str_array[1, 2] = options[JCTC.density_fitting]

    str_array[2, 1] = JCTC.contraction_mode
    str_array[2, 2] = options[JCTC.contraction_mode]

    str_array[3, 1] = JCTC.load
    str_array[3, 2] = options[JCTC.load]

    str_array[4, 1] = JCTC.guess
    str_array[4, 2] = options[JCTC.guess]

    str_array[5, 1] = JCTC.energy_convergence
    str_array[5, 2] = string(options[JCTC.energy_convergence])

    str_array[6, 1] = JCTC.density_convergence
    str_array[6, 2] = string(options[JCTC.density_convergence])

    str_array[7, 1] = JCTC.df_energy_convergence
    str_array[7, 2] = string(options[JCTC.df_energy_convergence])

    str_array[8, 1] = JCTC.df_density_convergence
    str_array[8, 2] = string(options[JCTC.df_density_convergence])

    str_array[9, 1] = JCTC.max_iterations
    str_array[9, 2] = string(options[JCTC.max_iterations])

    str_array[10, 1] = JCTC.df_max_iterations
    str_array[10, 2] = string(options[JCTC.df_max_iterations])

    str_array[11, 1] = JCTC.df_exchange_n_blocks
    str_array[11, 2] = string(options[JCTC.df_exchange_n_blocks])

    str_array[12, 1] = JCTC.df_screening_sigma
    str_array[12, 2] = string(options[JCTC.df_screening_sigma])

    str_array[13, 1] = JCTC.df_screen_exchange
    str_array[13, 2] = string(options[JCTC.df_screen_exchange])

    str_array[14, 1] = JCTC.df_force_dense
    str_array[14, 2] = string(options[JCTC.df_force_dense])

    str_array[15, 1] = JCTC.df_use_adaptive
    str_array[15, 2] = string(options[JCTC.df_use_adaptive])

    str_array[16, 1] = JCTC.num_devices
    str_array[16, 2] = string(options[JCTC.num_devices])

    str_array[17, 1] = JCTC.df_use_K_sym
    str_array[17, 2] = string(options[JCTC.df_use_K_sym])

    str_array[18, 1] = JCTC.df_K_sym_type
    str_array[18, 2] = string(options[JCTC.df_K_sym_type])

    write(file, file_name, str_array)
end


# function write_dictionary_to_hdf5{T}(file, dictionary::Dict{String, T}, file_name) where T
#     dict_keys = collect(keys(dictionary))
#     dict_values = collect(values(Dict))
#     write_arr = Array{String}(undef, length(dict_keys),2)
#     for i in eachindex(dict_keys)
#         write_arr[i, 1] = dict_keys[i]
#         write_arr[i, 2] = dict_values[i]
#     end
#     write(file, file_name, write_arr)
# end



function check_value_exists(key::String, dict::Dict{String, String}) :: String
    if haskey(dict, key)
        return dict[key]
    else
        return ""
    end
end

function save_run_level_data(file, jc_timing)
    str_array = Array{String}(undef, 12, 2)


    str_array[1, 1] = JCTC.run_name
    str_array[1, 2] = jc_timing.run_name

    str_array[2, 1] = JCTC.run_time
    str_array[2, 2] = string(jc_timing.run_time)

    str_array[3, 1] = JCTC.converged
    str_array[3, 2] = string(jc_timing.converged)

    str_array[4, 1] = JCTC.scf_energy
    str_array[4, 2] = string(jc_timing.scf_energy)

    str_array[5, 1] = JCTC.n_basis_functions
    str_array[5, 2] = check_value_exists(JCTC.n_basis_functions, jc_timing.non_timing_data)

    str_array[6, 1] = JCTC.n_auxiliary_basis_functions
    str_array[6, 2] = check_value_exists(JCTC.n_auxiliary_basis_functions, jc_timing.non_timing_data)

    str_array[7, 1] = JCTC.n_electrons
    str_array[7, 2] = check_value_exists(JCTC.n_electrons, jc_timing.non_timing_data)

    str_array[8, 1] = JCTC.n_occupied_orbitals
    str_array[8, 2] = check_value_exists(JCTC.n_occupied_orbitals, jc_timing.non_timing_data)

    str_array[9, 1] = JCTC.n_atoms
    str_array[9, 2] = check_value_exists(JCTC.n_atoms, jc_timing.non_timing_data)

    str_array[10, 1] = JCTC.n_threads
    str_array[10, 2] = check_value_exists(JCTC.n_threads, jc_timing.non_timing_data)

    str_array[11, 1] = JCTC.n_ranks
    str_array[11, 2] = check_value_exists(JCTC.n_ranks, jc_timing.non_timing_data)

    str_array[12, 1] = JCTC.n_iterations
    str_array[12, 2] = check_value_exists(JCTC.n_iterations, jc_timing.non_timing_data)
    

    write(file, JCTC.run_level_data , str_array)
end


# using Test
# function check_run_level_data(data::Array{String, 2})

#     #check the keys are correct
#     @test data[1, 1] == JCTC.run_name
#     @test data[2, 1] == JCTC.run_time
#     @test data[3, 1] == JCTC.converged
#     @test data[4, 1] == JCTC.scf_energy
#     @test data[5, 1] == JCTC.n_basis_functions
#     @test data[6, 1] == JCTC.n_auxiliary_basis_functions
#     @test data[7, 1] == JCTC.n_electrons
#     @test data[8, 1] == JCTC.n_occupied_orbitals
#     @test data[9, 1] == JCTC.n_atoms
#     @test data[10, 1] == JCTC.n_threads
#     @test data[11, 1] == JCTC.n_ranks

#     #check the values are correct 
#     @test data[1, 2] == "test"
#     @test data[2, 2] == "1.0"
#     @test data[3, 2] == "true"
#     @test data[4, 2] == "-1.0"
#     @test data[5, 2] == "100"
#     @test data[6, 2] == "1000"
#     @test data[7, 2] == "10"
#     @test data[8, 2] == "5"
#     @test data[9, 2] == "4"
#     @test data[10, 2] == "5"
#     @test data[11, 2] == "1"
# end

# function set_test_run_level_data(jc_timing)
#     jc_timing.run_time = 1.0
#     jc_timing.run_name = "test"
#     jc_timing.converged = true
#     jc_timing.scf_energy = -1.0
#     jc_timing.non_timing_data[JCTC.n_basis_functions] = "100"
#     jc_timing.non_timing_data[JCTC.n_auxiliary_basis_functions] = "1000"
#     jc_timing.non_timing_data[JCTC.n_electrons] = "10"
#     jc_timing.non_timing_data[JCTC.n_occupied_orbitals] = "5"
#     jc_timing.non_timing_data[JCTC.n_atoms] = "4"
#     jc_timing.non_timing_data[JCTC.n_threads] = "5"
#     jc_timing.non_timing_data[JCTC.n_ranks] = "1"

#     jc_timing.non_timing_data[JCTC.screened_indices_count] = "10000"
#     jc_timing.non_timing_data[JCTC.unscreened_exchange_blocks] = "10"
#     jc_timing.non_timing_data[JCTC.n_iterations] = "1.0"

    
#     jc_timing.non_timing_data["GPU_num_devices"] = "1"
#     jc_timing.non_timing_data["GPU_data_size_MB"] = "1000"
# end


# function set_test_iteration_data(jc_timing)
#     jc_timing.timings[JCTC.H_time] = 1.0
#     jc_timing.timings[JCTC.screening_time] = 2.0
#     jc_timing.timings[JCTC.screening_metadata_time] = 3.0
#     jc_timing.timings[JCTC.three_eri_time] = 4.0
#     jc_timing.timings[JCTC.two_eri_time] = 5.0
#     jc_timing.timings[JCTC.B_time] = 6.0

    
#     for i in 1:10
#         jc_timing.timings[JCTiming_key(JCTC.iteration_time, i)] = i*1000 
#         jc_timing.timings[JCTiming_key(JCTC.fock_time, i)] = i*1000 + 1
#         jc_timing.timings[JCTiming_key(JCTC.density_time, i)] = i*1000 + 2
#         jc_timing.timings[JCTiming_key(JCTC.H_add_time, i)] = i*1000 + 3
#         jc_timing.timings[JCTiming_key(JCTC.K_time, i)] = i*1000 + 4
#         jc_timing.timings[JCTiming_key(JCTC.W_time, i)] = i*1000 + 5
#         jc_timing.timings[JCTiming_key(JCTC.J_time, i)] = i*1000 + 6
#         jc_timing.timings[JCTiming_key(JCTC.V_time, i)] = i*1000 + 7


#         jc_timing.timings[JCTiming_key(JCTC.form_J_AB_inv_time, i)] = i*1000 + 9
#     end
# end

# function check_timing_values(data)
#     # convert data back to string float dict 
#     data_dict = Dict{String, Float64}()
#     #iterate over the rows
#     for i in axes(data, 1)
#         data_dict[data[i, 1]] = parse(Float64, data[i, 2])
#     end

#     #check the run level values
#     @test data_dict[JCTC.H_time] == 1.0
#     @test data_dict[JCTC.screening_time] == 2.0
#     @test data_dict[JCTC.screening_metadata_time] == 3.0
#     @test data_dict[JCTC.three_eri_time] == 4.0
#     @test data_dict[JCTC.two_eri_time] == 5.0
#     @test data_dict[JCTC.B_time] == 6.0

#     #check the iteration values
#     for iter in 1:10
#         @test data_dict[JCTiming_key(JCTC.iteration_time, iter)] == iter*1000
#         @test data_dict[JCTiming_key(JCTC.fock_time, iter)] == iter*1000 + 1
#         @test data_dict[JCTiming_key(JCTC.density_time, iter)] == iter*1000 + 2
#         @test data_dict[JCTiming_key(JCTC.H_add_time, iter)] == iter*1000 + 3
#         @test data_dict[JCTiming_key(JCTC.K_time, iter)] == iter*1000 + 4
#         @test data_dict[JCTiming_key(JCTC.W_time, iter)] == iter*1000 + 5
#         @test data_dict[JCTiming_key(JCTC.J_time, iter)] == iter*1000 + 6
#         @test data_dict[JCTiming_key(JCTC.V_time, iter)] == iter*1000 + 7
#         @test data_dict[JCTiming_key(JCTC.form_J_AB_inv_time, iter)] == iter*1000 + 9
#     end

# end

# function set_scf_options(jc_timing)
#     jc_timing.options[JCTC.density_fitting] = "true"
#     jc_timing.options[JCTC.contraction_mode] = "dense"
#     jc_timing.options[JCTC.load] = "false"
#     jc_timing.options[JCTC.guess] = "sad"
#     jc_timing.options[JCTC.energy_convergence] = "1e-6"
#     jc_timing.options[JCTC.density_convergence] = "1e-6"
#     jc_timing.options[JCTC.df_energy_convergence] = "1e-6"
#     jc_timing.options[JCTC.df_density_convergence] = "1e-6"
#     jc_timing.options[JCTC.max_iterations] = "100"
#     jc_timing.options[JCTC.df_max_iterations] = "100"
#     jc_timing.options[JCTC.df_exchange_n_blocks] = "100"
#     jc_timing.options[JCTC.df_screening_sigma] = "1e-6"
#     jc_timing.options[JCTC.df_screen_exchange] = "true"
# end

# function check_scf_options(data)
#     #convert array back to dict{string, string}
#     data_dict = Dict{String, String}()
#     for i in axes(data, 1)
#         data_dict[data[i, 1]] = data[i, 2]
#     end

#     @test data_dict[JCTC.density_fitting] == "true"
#     @test data_dict[JCTC.contraction_mode] == "dense"
#     @test data_dict[JCTC.load] == "false"
#     @test data_dict[JCTC.guess] == "sad"
#     @test data_dict[JCTC.energy_convergence] == "1e-6"
#     @test data_dict[JCTC.density_convergence] == "1e-6"
#     @test data_dict[JCTC.df_energy_convergence] == "1e-6"
#     @test data_dict[JCTC.df_density_convergence] == "1e-6"
#     @test data_dict[JCTC.max_iterations] == "100"
#     @test data_dict[JCTC.df_max_iterations] == "100"
#     @test data_dict[JCTC.df_exchange_n_blocks] == "100"
#     @test data_dict[JCTC.df_screening_sigma] == "1e-6"
#     @test data_dict[JCTC.df_screen_exchange] == "true"
    
# end

# function set_non_timing_data(jc_timing)
#     jc_timing.non_timing_data[JCTC.screened_indices_count] = "10000"
#     jc_timing.non_timing_data[JCTC.unscreened_exchange_blocks] = "10"
#     jc_timing.non_timing_data[JCTC.n_iterations] = "10"
#     num_devices = 4
#     jc_timing.non_timing_data[JCTC.GPU_num_devices] = string(num_devices)
#     for device_id in 1:num_devices
#         jc_timing.non_timing_data[JCTiming_GPUkey(JCTC.GPU_data_size_MB, device_id)] = string(1000*device_id)
#     end

# end

# function check_non_timing_data(data)
#     #convert array back to dict{string, string}
#     data_dict = Dict{String, String}()
#     for i in axes(data, 1)
#         data_dict[data[i, 1]] = data[i, 2]
#     end

#     @test data_dict[JCTC.screened_indices_count] == "10000"
#     @test data_dict[JCTC.unscreened_exchange_blocks] == "10"
#     @test data_dict[JCTC.n_iterations] == "10"
#     @test data_dict[JCTC.GPU_num_devices] == "4"
#     num_devices = parse(Int64, data_dict[JCTC.GPU_num_devices])
#     for device_id in 1:num_devices
#         @test data_dict[JCTiming_GPUkey(JCTC.GPU_data_size_MB, device_id)] == string(1000*device_id)
#     end

# end

# function test()
#     jc_timing = create_jctiming()

#     set_test_run_level_data(jc_timing)
#     set_test_iteration_data(jc_timing)
#     set_scf_options(jc_timing)
#     set_non_timing_data(jc_timing)

#     save_jc_timings_to_hdf5(jc_timing, "/pscratch/sd/j/jhayes1/source/JuliaChem.jl/testoutputs/test_jc_timings.h5")

#     # read in the data
#     h5open("/pscratch/sd/j/jhayes1/source/JuliaChem.jl/testoutputs/test_jc_timings.h5", "r") do file
#         data = read(file, JCTC.run_level_data)
#         check_run_level_data(data)
#         data = read(file, JCTC.timings)
#         check_timing_values(data)
#         data = read(file, JCTC.scf_options)
#         check_scf_options(data)
#         data = read(file, JCTC.non_timing_data)
#         check_non_timing_data(data)
#         println("all tests passed")
#     end

# end

# test()
