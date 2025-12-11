include("../benchmark.jl")
using Serialization

function run_test(input_file_path:: String, results_file_output::String, number_of_samples::Int, basis, aux_basis, startup_file_path)
    println("Running RHF vs DF-RHF vs DF guess RHF test: $input_file_path")
    # run to compile ignore timings for these
    test_results = Dict()
    
    DF_RHF_BLAS = "DF_RHF_HCORE"
    DF_RHF_TENOP = "DF_RHF_TENOP"
    RHF_HCORE = "RHF_HCORE"

    test_results[DF_RHF_BLAS] = Dict()
    test_results[DF_RHF_TENOP] = Dict()
    test_results[RHF_HCORE] = Dict()

    guess = "hcore"
    contraction_mode_TO = "TensorOperations"
    contraction_mode_BLAS = "BLAS"
    static_load = "static"
    start_up_basis = "6-31G"
    start_up_aux_basis = "cc-pVTZ-JKFIT"

    for iter in 1:number_of_samples
        run_df_rhf(startup_file_path, start_up_basis ,start_up_aux_basis, false, guess, contraction_mode_TO, static_load)
        do_named_run_dfrhf(test_results, iter, DF_RHF_BLAS, input_file_path, basis, aux_basis, guess, false, contraction_mode_BLAS, static_load)
        run_df_rhf(startup_file_path, start_up_basis ,start_up_aux_basis, false, guess, contraction_mode_BLAS, static_load)
        do_named_run_dfrhf(test_results, iter, DF_RHF_TENOP, input_file_path, basis, aux_basis, guess, false, contraction_mode_TO, static_load)
        run_rhf(startup_file_path, start_up_basis, guess, static_load)
        do_named_run_rhf(test_results, iter, RHF_HCORE, input_file_path, basis, guess, static_load)
    end

    serialize(results_file_output, test_results)
end

function do_named_run_rhf(test_results, iter, name, input_file_path, basis,  guess, load)
    println("Running $name test: $iter")
    results = run_rhf(input_file_path, basis, guess, load)
    collect_run_results(test_results, name, results, iter)
    flush(stdout)

end

function do_named_run_dfrhf(test_results, iter, name, input_file_path, basis, aux_basis ,guess, df_is_guess, contraction_mode, load)
    println("Running $name test: $iter")
    results = run_df_rhf(input_file_path, basis, aux_basis, df_is_guess, guess, contraction_mode, load)
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

    if typeof(rhf_results.Basis) == JuliaChem.JCBasis.CalculationBasisSets
        results_dict["Basis_Count"] =  rhf_results.Basis.primary.norb
        results_dict["Aux_Basis_Count"] = rhf_results.Basis.auxillary.norb
    else
        results_dict["Basis_Count"] = rhf_results.Basis.norb
    end
    results[run_name][iter] = results_dict
end

function get_file_paths(inputs_dir)
    S22_3_directory = joinpath(@__DIR__, inputs_dir)
    input_file_paths = readdir(S22_3_directory)
    input_file_paths .= S22_3_directory .* input_file_paths
    return input_file_paths 
end


number_of_samples = 3

inputs_dir = "../../../S22_3/6-311++G_2d_2p/"
startup_file_path = "../start_up.json"
startup_df_file_path = "../start_up_df.json"

startup_file_path = joinpath(@__DIR__, startup_file_path)
startup_df_file_path = joinpath(@__DIR__, startup_df_file_path)

input_file_paths = get_file_paths(inputs_dir)

println(input_file_paths)

JuliaChem.initialize()

for file in input_file_paths
    file_name = split(file, "/")[end]
    file_name = split(file_name, ".")[1]
    run_test(file, "./results/$(file_name)_results.data", number_of_samples,"6-311++G(2d,2p)", "aug-cc-pVTZ-JKFIT", startup_file_path)
end

# for s22_number in [7 15 21]
#     output_file_path = "./results/S22_1NODE_36threads_results_$(s22_number)_static.data"
#     run_s22_test_BLAS_VS_TensorOp(inputs_dir, s22_number, output_file_path, number_of_samples)
# end
JuliaChem.finalize()

