using Serialization
using StatsPlots 
using Random
using Plots.PlotMeasures
using DataFrames 
using CSV

# RHF_HCORE = "RHF_HCORE"
# DF_RHF_HCORE = "DF_RHF_HCORE"
# DF_GUESS_RHF_HCORE = "DF_GUESS_RHF_HCORE"
# DF_RHF_HCORE_TENOP = "DF_RHF_HCORE_TENOP"
# DF_GUESS_RHF_HCORE_TENOP = "DF_GUESS_RHF_HCORE_TENOP"

# RHF_SAD = "RHF_SAD"
# DF_RHF_SAD = "DF_RHF_SAD"

function get_result_run_energy(results, run_name)
    run = results[run_name]
    first_run_energy :: Float64 = run[1]["Energy"]
    for run_iteration_key in keys(run)
        if !isapprox(run[run_iteration_key]["Energy"], first_run_energy)
            println("ERROR: Energies are not the same for $(run_name)")
            println("Energy #$(run_iteration_key): $(run[run_iteration_key]["Energy"])")
            println("First Energy: $(first_run_energy)")
        end
    end
    return first_run_energy

end

function calculate_energy_ΔE(results, run_name1, run_name2)

    run_name1_energy = get_result_run_energy(results, run_name1)
    run_name2_energy = get_result_run_energy(results, run_name2)
    return run_name1_energy - run_name2_energy
end

function validate_results(test_output_directory)
    S22_directory = joinpath(@__DIR__, test_output_directory)
    output_files = readdir(S22_directory)
    output_file_paths = S22_directory .* output_files

    sort!(output_file_paths)

    for output_file in output_file_paths
        if occursin(".log", output_file)
            continue
        end
        s22_number = parse(Int, split(split(output_file, "/")[end], ".")[1])
        println()
        println("displaying Results for $(output_file)")
        compare_results_for_file(output_file, s22_number)
    end
end

function print_run_total_energies(run_name, run)
    iterations = collect(keys(run))
    run_energies = []
    for iteration in iterations
        push!(run_energies, run[iteration]["Energy"])
    end
    println("$(run_name) Energies: $(run_energies)")
end

function print_run_times(run_name, run)
    iterations = collect(keys(run))
    run_times = []
    for iteration in iterations
        push!(run_times, run[iteration]["Run_Time"])
    end
    println("$(run_name) Times: $(run_times)")
end

function print_run_iteration_counts(run_name, run)
    iterations = collect(keys(run))
    run_iteration_counts = []
    for iteration in iterations
        scf_iterations = run[iteration]["Iteration_Times"]
        push!(run_iteration_counts, length(scf_iterations))
    end
    println("$(run_name) Iteration Counts: $(run_iteration_counts)")
end

function fix_rhf_sad(results)

    bundled_results = results[RHF_SAD]
    println("$(collect(keys(bundled_results)))")
end

function compare_runs(results, run_name_1, run_name_2)
    println()
    println("-----------------------")
    ΔE = calculate_energy_ΔE(results, run_name_1, run_name_2)
    println("ΔE $(run_name_1) vs $(run_name_2): $ΔE")

    speedup = calculate_speedup(results, run_name_1, run_name_2)
    println("Speedup $(run_name_1) vs $(run_name_2): $speedup")

    run1_average_df_time, run_1_ave_first_iteration_df_time, run_1av_iteration_time =
     calculate_iteration_times(results, run_name_1)

    println("$(run_name_1) times: ")
    println("Average First Iteration Time : $(run1_average_df_time)")
    println("Average DF Iteration Time: $(run_1_ave_first_iteration_df_time)")
    println("Average RHF Iteration Time: $(run_1av_iteration_time)")

    run2_average_df_time, run_2_ave_first_iteration_df_time, run2_av_iteration_time =
     calculate_iteration_times(results, run_name_2)

    println("$(run_name_2) times: ")
    println("Average First Iteration Time : $(run2_average_df_time)")
    println("Average DF Iteration Time: $(run_2_ave_first_iteration_df_time)")
    println("Average RHF Iteration Time: $(run2_av_iteration_time)")
    println("-----------------------")
    println()

end

function compare_results_for_file(results_data_file_path)
    results = deserialize(results_data_file_path)
    all_keys = keys(results)
    tolerance = .001
    any_exceed_tolerance = false
    for key in keys(results)
        for key2 in all_keys
            if key != key2
                ΔE = calculate_energy_ΔE(results, key, key2)
                if ΔE > tolerance
                    println("ERROR ΔE for $key and $key2 on file $results_data_file_path > than tolerance $tolerance")
                    any_exceed_tolerance = true
                end
            end
        end
    end    
    if !any_exceed_tolerance 
        println("all ΔE tolereances met for $results_data_file_path")
    end
end

function calculate_iteration_times(results, run_name)
    run = results[run_name]
    
    df_iteration_times = 0.0
    first_iteration_df_time = 0.0
    rhf_iteration_time = 0.0

    number_of_first_iterations = 0
    number_of_iterations = 0
    number_of_rhf_iterations = 0

    run_iteration_keys = collect(keys(run))
    density_fitting = false
    for run_iteration in run_iteration_keys
        run_iteration = run[run_iteration]
        df_iterations_start = run_iteration["density_fitting_iteration_range_start"]
        df_iterations_end = run_iteration["density_fitting_iteration_range_end"]
        timings = run_iteration["Iteration_Times"]
        timings_keys = collect(keys(timings))
        density_fitting = df_iterations_start > 0
        for scf_iteration_key in timings_keys 
            iteration_index = tryparse(Int, scf_iteration_key)
            if isnothing(iteration_index) 
                continue
            end

            iteration_time = timings[scf_iteration_key]
            if density_fitting && iteration_index == df_iterations_start 
                first_iteration_df_time += iteration_time
                number_of_first_iterations += 1
            elseif density_fitting && iteration_index < df_iterations_end
                df_iteration_times += iteration_time
                number_of_iterations += 1
            else 
                rhf_iteration_time += iteration_time
                number_of_rhf_iterations += 1
            end
        end
        
    end   

    average_df_time = density_fitting ? df_iteration_times / number_of_iterations : 0.0
    av_first_iteration_df_time = density_fitting ? first_iteration_df_time / number_of_first_iterations : 0.0 
    rhf_average_iteration_time = rhf_iteration_time / number_of_rhf_iterations

    return average_df_time, av_first_iteration_df_time, rhf_average_iteration_time
end


function calculate_average_runtime(results, run_name)
    iteration_keys = collect(keys(results[run_name]))
    total_run_time = 0.0
    for iteration in iteration_keys
        total_run_time += results[run_name][iteration]["Run_Time"]
    end
    number_of_iterations = length(iteration_keys)
    average_runtime = total_run_time / number_of_iterations
    return average_runtime
end

function calculat_average_iteration_one(results, run_keys)
    run_iteration_1_times = []
    i = 1
    for run_key in run_keys
        iteration_one_time = 0.0
        iteration_keys = collect(keys(results[run_key]))
        for iteration in iteration_keys
            iteration_one_time += results[run_key][iteration]["Iteration_Times"]["1"]
        end
        push!(run_iteration_1_times, iteration_one_time / length(iteration_keys))
    end
    
    return run_iteration_1_times
end

function get_runtimes(results, run_name)
    iteration_keys = collect(keys(results[run_name]))
    run_times = []
    for iteration in iteration_keys
        push!(run_times, results[run_name][iteration]["Run_Time"])
    end
    return run_times
end

function get_number_of_basis_functions(results, rhf_key, df_key)

    basis_count = results[rhf_key][1]["Basis_Count"]
    aux_basis_count = results[df_key][1]["Aux_Basis_Count"]
    return basis_count, aux_basis_count
end

function plot_s22_vs_rhf_hcore()
    original_avg_run_times = zeros(22)
    avg_run_times = zeros(22,3)

    run_names = [DF_RHF_HCORE, DF_GUESS_RHF_HCORE , DF_RHF_HCORE_TENOP]
    for i in 1:22
        results =  deserialize("./testoutputs/DF-VS-RHF_S22/df_vs_rhf_same_etol/$i.data")
        original_avg_run_times[i] = calculate_average_runtime(results, RHF_HCORE)
        avg_run_times[i,1] = calculate_average_runtime(results, DF_GUESS_RHF_HCORE)
        avg_run_times[i,2] = calculate_average_runtime(results, DF_RHF_HCORE)
        avg_run_times[i,3] = calculate_average_runtime(results, DF_RHF_HCORE_TENOP)
    end
    plot_s22_speedup_results(original_avg_run_times, RHF_HCORE, avg_run_times, run_names)
end

function plot_s22_speedup_results(original_run_times, original_run_name, run_times, run_time_names; title="", filename="")
    
    speedups = original_run_times ./ run_times
    number_of_groups = size(run_times)[2]
    number_of_files = size(run_times)[1]
    run_time_names_clean = replace.(run_time_names, "_" => " ")
    original_name_cleaned_up = replace(original_run_name, "_" => " ")
    groups = repeat(run_time_names_clean, inner=number_of_files)
    name = repeat(1:22, outer=number_of_groups)
    if title == ""
        title = "Speedups vs $(original_name_cleaned_up)"
    end
    bar_colors = reshape([:black, :green, :red, :blue, :orange, :purple], (1,6))
    groupedbar(name, speedups, group = groups, xlabel = "S22 Test", ylabel = "Speedup",
        title = "Speedups vs $(original_name_cleaned_up)",
        bar_width = .8,
        lw = 0, framestyle = :box,  legend = :topleft, color= bar_colors)
    hline!([1], line=(2, :dash, 1.0, [:black]), label="")
    plot!(size=(800,600))
    plot!(xticks= 1:22)
    plot!(yticks= 1:2:12)
    plot!(xlim=(0.5,22.5))
    plot!(margin=20px)
    savefig("$(filename).png")
end 


function plot_df_algorithm_speedups(original_run_key, avg_run_times, run_keys,run_names, file_name_prefix, plot_title)
    original_index = findfirst(x->x == original_run_key, run_keys)
    df_avg_run_times = avg_run_times[:, 1:end .!= original_index]
    df_run_names = run_names[1:end .!= original_index]

    number_of_files = length(avg_run_times[:,1])
    number_of_groups = size(df_avg_run_times)[2]
    speedups = avg_run_times[:,original_index] ./ df_avg_run_times




    groups = repeat(df_run_names, inner=number_of_files)
    name = repeat(1:number_of_files, outer=number_of_groups)
    if plot_title == ""
        plot_title = "Density Fitting Speedups vs RHF"
    end
    bar_colors = reshape([:blue, :gray, :green, :red, :orange, :purple], (1,6))
    groupedbar(name, speedups, group = groups, xlabel = "S22 Test", ylabel = "Speed up",
        title = plot_title,
        bar_width = .8,
        lw = 0, framestyle = :box,  legend = :topleft, color= bar_colors)
    plot!(size=(1000,600))
    plot!(xticks= 1:22)
    plot!(yticks= 1:2:12)
    plot!(margin=30px)
    savefig("$(file_name_prefix)_DF_Speedup.png")

end

function create_avg_runtime_csv(avg_run_times, run_names, file_name_prefix)
    data = DataFrame()
    data[:,"S22"] = 1:length(avg_run_times[:,1])
    i = 1
    for name in run_names
        data[:,name] = avg_run_times[:,i]
        i+= 1
    end
    CSV.write("$(file_name_prefix)_avg.csv", data)
end

function create_iteration_one_csv(avg_iteration_one_times, run_names, file_name_prefix)
    data = DataFrame()
    data[:,"S22"] = 1:length(avg_iteration_one_times[:,1])
    i = 1
    for name in run_names
        data[:,name] = avg_iteration_one_times[:,i]
        i+= 1
    end
    CSV.write("$(file_name_prefix)_avg_iteration_one.csv", data)
end

function create_basis_count_csv(number_of_basis_functions, number_of_aux_basis_functions, file_name_prefix)
    data = DataFrame()
    data[:,"S22"] = 1:length(number_of_basis_functions)
    data[:,"Basis Count"] = number_of_basis_functions
    data[:,"Aux Basis Count"] = number_of_aux_basis_functions
    CSV.write("$(file_name_prefix)_basis_func_count.csv", data)
end

function create_speedup_csv(original_run_key, original_run_name, avg_run_times, run_names, file_name_prefix)
    original_index = findfirst(x->x == original_run_key, run_keys)
    original_avg_run_times = avg_run_times[:,original_index]

    data = DataFrame()
    data[:,"S22"] = 1:length(original_avg_run_times)
    i = 1
    for name in run_names
        data[:, name] = round.(original_avg_run_times ./ avg_run_times[:,i], digits=2)
        i+= 1
    end
    CSV.write("$(file_name_prefix)_speedup.csv", data)
end

function create_Δ_E_csv(original_run_key, original_run_name, energies, run_keys, run_names, file_name_prefix)
    original_index = findfirst(x->x == original_run_key, run_keys) 
    original_energies = energies[:,original_index]
    data = DataFrame()
    data[:,"S22"] = 1:length(original_energies)
    data[:,"$original_run_name Energy"] = original_energies
    i=1
    for run_key in run_keys
        if run_key != original_run_key
            data[:, "ΔE $(run_names[i])"] = original_energies .- energies[:,i]
        end        
        i+=1
    end
    CSV.write("$(file_name_prefix)_DeltaE.csv", data)
end

function create_run_time_csv(run_times, run_keys, run_names, file_name_prefix)
    data = DataFrame()

    i=1
    for run_key in run_keys
        data[:, "$(run_names[i])"] = run_times[i,:]
        i+=1
    end
    CSV.write("$(file_name_prefix)_runtimes.csv", data)

end


function process_s22_static_results(run_keys, run_names, run_file_names, rhf_run_key, rhf_run_name, df_key, result_file_prefix, number_of_samples)
    number_of_runs = length(run_keys)
    number_of_files = length(run_file_names)
    
    original_avg_run_times = zeros(number_of_files)
    avg_run_times = zeros(number_of_files,number_of_runs)
    avg_iteration_one_times = zeros(number_of_files,number_of_runs)

    original_energies = zeros(number_of_files)
    energies = zeros(number_of_files,number_of_runs)
    number_of_basis_functions = zeros(Int64, number_of_files)
    number_of_aux_basis_functions = zeros(Int64, number_of_files)

    file_index = 1
    file_results = []
    for path in run_file_names

        results =  deserialize(path)
        push!(file_results,results)
        run_times = zeros(number_of_runs, number_of_samples)

        basis_count, aux_basis_count = get_number_of_basis_functions(results, rhf_run_key, df_key)
        number_of_basis_functions[file_index] = basis_count
        number_of_aux_basis_functions[file_index] = aux_basis_count

        run_index = 1

        avg_iteration_one_times[file_index, :] = calculat_average_iteration_one(results, run_keys)


        for run_key in run_keys
            run_times[run_index, :] .= get_runtimes(results, run_key)
            avg_run_times[file_index,run_index] = calculate_average_runtime(results, run_key)
            
            energies[file_index,run_index] = get_result_run_energy(results, run_key)

            run_index += 1
        end
        file_prefix = "$(result_file_prefix)_$(file_index)" 
        create_run_time_csv(run_times, run_keys, run_names, file_prefix)
        file_index += 1
    end
    
    create_Δ_E_csv(rhf_run_key, rhf_run_name, energies, run_keys, run_names, result_file_prefix)
    create_avg_runtime_csv(avg_run_times, run_names, result_file_prefix)
    create_iteration_one_csv(avg_iteration_one_times, run_names, result_file_prefix)
    create_basis_count_csv(number_of_basis_functions, number_of_aux_basis_functions, result_file_prefix)
    plot_df_algorithm_speedups(rhf_run_key, avg_run_times, run_keys,run_names, result_file_prefix, "Density Fitting Speedups vs RHF")
    
    # create_speedup_csv(original_avg_run_times, RHF_HCORE , avg_run_times, run_names, filename)
    # plot_s22_speedup_results(original_avg_run_times, RHF_HCORE, avg_run_times, run_names; filename =filename)
end

function get_s22_scaling_results(path, s22_numbers, thread_count_runs)
    for s22_number in s22_numbers 
        average_times_DF = Dict{Int, Float64}()
        average_times_DF_Tenop = Dict{Int, Float64}()
        for run_thread_count in thread_count_runs
            results = deserialize("$path/$run_thread_count/$s22_number.data")
            average_times_DF[run_thread_count] = calculate_average_runtime(results, DF_RHF_HCORE)
            average_times_DF_Tenop[run_thread_count] = calculate_average_runtime(results, DF_RHF_HCORE_TENOP)
        end    
        println("Times for s22: ", s22_number)    
        println("DF Hcore: ", average_times_DF)
        println("DF Hcore T.O.: ", average_times_DF_Tenop)
    end
end 
# get_s22_results("/home/jackson/source/JuliaChem.jl/testoutputs/DF-VS-RHF_S22/df_vs_rhf_s22_memcleanup_2")
# get_s22_scaling_results("./testoutputs/DF-VS-RHF_S22/df_vs_rhf_thread_scaling" , [8 1 4 6], [2 4 8 16])

# plot_s22_vs_rhf_hcore()
# plot_s22_tenop_vs_hcore()

# validate_results("/home/jackson/source/JuliaChem.jl/testoutputs/DF-VS-RHF_S22/df_vs_rhf_same_etol_mem_cleanup/")
# compare_results_for_file("/home/jackson/source/JuliaChem.jl/S22_results_5_after_mem_cleanup.data")

function process_s22_static()
    DF_RHF_BLAS = "DF_RHF_HCORE"
    DF_RHF_TENOP = "DF_RHF_TENOP"
    RHF_HCORE = "RHF_HCORE"
    RHF_run_name = "RHF"
    run_keys = [DF_RHF_BLAS, DF_RHF_TENOP, RHF_HCORE]
    run_names = ["DF BLAS", "DF Tensor", RHF_run_name]
    run_file_names = []
    path_to_results = joinpath(@__DIR__, "./results")
    for s22_number in 1:22
        output_file_path = "$path_to_results/S22_1NODE_36threads_results_$(s22_number)_static.data"
        push!(run_file_names, output_file_path)
        compare_results_for_file(output_file_path)
    end

    result_file_prefix = "S22_36Threads_Static_HASWELL_BOLT_run2"
    number_of_samples = 3
    process_s22_static_results(run_keys, run_names, run_file_names, RHF_HCORE, RHF_run_name, DF_RHF_BLAS, result_file_prefix, number_of_samples)

end

process_s22_static()
