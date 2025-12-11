include("../tools/travis/travis-rhf.jl")
using Serialization
function run_s22_test(s22_number, output_file_path, number_of_samples)
    #== select input files ==#
    S22_directory = joinpath(@__DIR__, "../example_inputs/S22/")
    inputs = readdir(S22_directory)
    inputs .= S22_directory .* inputs

    input = inputs[s22_number]
    
    JuliaChem.initialize()
    
    run_test(input, output_file_path, number_of_samples, s22_number)

    JuliaChem.finalize()
end

function run_test(input_file_path:: String, results_file_output::String, number_of_samples::Int, s22_number)
    println("Running RHF vs DF-RHF vs DF guess RHF test: $input_file_path")
    # run to compile ignore timings for these
    test_results = Dict()
    
    RHF_HCORE = "RHF_HCORE"
    DF_RHF_HCORE = "DF_RHF_HCORE"
    DF_GUESS_RHF_HCORE = "DF_GUESS_RHF_HCORE"
    DF_RHF_HCORE_TENOP = "DF_RHF_HCORE_TENOP"
    DF_GUESS_RHF_HCORE_TENOP = "DF_GUESS_RHF_HCORE_TENOP"

    RHF_SAD = "RHF_SAD"
    DF_RHF_SAD = "DF_RHF_SAD"

    test_results[RHF_HCORE] = Dict()
    test_results[DF_RHF_HCORE] = Dict()
    test_results[DF_GUESS_RHF_HCORE] = Dict()
    test_results[DF_RHF_HCORE_TENOP] = Dict()
    test_results[DF_GUESS_RHF_HCORE_TENOP] = Dict()

    test_results[RHF_SAD] = Dict()
    test_results[DF_RHF_SAD] = Dict()

    guess = "hcore"
    # run to compile ignore timings for these
    # travis_rhf(input_file_path, guess)
    contraction_mode = "TensorOperations"
    travis_rhf_density_fitting(input_file_path, "cc-pVTZ-JKFIT", true, guess, contraction_mode)
    contraction_mode = "default"
    travis_rhf_density_fitting(input_file_path, "cc-pVTZ-JKFIT", false, guess, contraction_mode)

    println("doing s22 = $s22_number")
    for iter in 1:number_of_samples
        guess = "hcore"
        # do_named_run_rhf(test_results, iter, RHF_HCORE, input_file_path, guess)
        contraction_mode = "default"
        do_named_run_dfrhf(test_results, iter, DF_RHF_HCORE, input_file_path, guess, false, contraction_mode)
        # do_named_run_dfrhf(test_results, iter, DF_GUESS_RHF_HCORE, input_file_path, guess, true, contraction_mode)
        contraction_mode = "TensorOperations"
        do_named_run_dfrhf(test_results, iter, DF_RHF_HCORE_TENOP, input_file_path, guess, false, contraction_mode)
        # do_named_run_dfrhf(test_results, iter, DF_GUESS_RHF_HCORE_TENOP, input_file_path, guess, true, contraction_mode)
        # guess = "sad"
        # do_named_run_rhf(test_results, iter, RHF_SAD, input_file_path, guess)
        # do_named_run_dfrhf(test_results, iter, DF_RHF_SAD, input_file_path, guess, false, contraction_mode)
    end

    serialize(results_file_output, test_results)
end

function do_named_run_rhf(test_results, iter, name, input_file_path, guess)
    println("Running $name test: $iter")
    results = travis_rhf(input_file_path, guess)
    collect_run_results(test_results, name, results, iter)
    flush(stdout)

end

function do_named_run_dfrhf(test_results, iter, name, input_file_path, guess, df_is_guess, contraction_mode)
    println("Running $name test: $iter")
    results = travis_rhf_density_fitting(input_file_path, "cc-pVTZ-JKFIT", df_is_guess, guess, contraction_mode)
    collect_run_results(test_results, name, results, iter)
    flush(stdout)
end

function collect_run_results(results, run_name, rhf_results, iter)
    results_dict = Dict()
    results_dict["Keywords"] = rhf_results.Keywords
    results_dict["Molecule"] = rhf_results.Molecule
    results_dict["Model"] = rhf_results.Model
    results_dict["Energy"] = rhf_results.Energy["Energy"]
    results_dict["Iteration_Times"] = rhf_results.Energy["Timings"].iteration_times
    results_dict["Run_Time"] = rhf_results.Energy["Timings"].run_time
    results_dict["density_fitting_iteration_range_start"] = rhf_results.Energy["Timings"].density_fitting_iteration_range[1]
    results_dict["density_fitting_iteration_range_end"] = rhf_results.Energy["Timings"].density_fitting_iteration_range[end]
    results_dict["Properties"] = rhf_results.Properties
    results[run_name][iter] = results_dict
   
    
end



args_length = length(ARGS)
s22_number = 5
output_file_path = "./S22_results_$(s22_number)_after_mem_cleanup.data"
number_of_samples = 3

if args_length >= 1
    s22_number = parse(Int, ARGS[1])
end
if args_length >= 2
    output_file_path = ARGS[2]
end
if args_length >= 3
    number_of_samples = parse(Int, ARGS[3])
end

run_s22_test(s22_number, output_file_path, number_of_samples)

