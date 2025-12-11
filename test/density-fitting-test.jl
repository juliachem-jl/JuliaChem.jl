#=============================#
#== put needed modules here ==#
#=============================#
ENV["MKL_DYNAMIC"] = false
using MKL
println("starting density fitting test"); flush(stdout)
using JuliaChem
println("imported JuliaChem"); flush(stdout)
using Test
using JuliaChem.Shared
using JuliaChem.Shared.JCTC

using MPI
using LinearAlgebra
using Base.Threads
# using ThreadPinning
using CUDA  

include("../example_scripts/full-rhf-repl.jl")
include("../example_scripts/save_jc_timings.jl")

#==================================================================
 Script to check if the Density fitted method 
 values are close to the ones produced by non density fitted RHF
==================================================================#

function check_density_fitted_method_matches_RHF(denity_fitted_input_file::String, input_file::String, output_path::String,run_warmup=true)
  # try 

    println("running density fitted file $denity_fitted_input_file")

    outputval = 3

    #startup compilation runs
   # screened_cpu_time = @elapsed begin @time begin 
    #   screened_scf_results, screened_properties = full_rhf(input_file)    # screened cpu df-rhf
    # end end
    # if run_warmup
      println("starting warm up")
      df_scf_results, density_fitted_properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted_gpu.json"), output=outputval)
      run_time = @elapsed df_scf_results, density_fitted_properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted_gpu.json"), output=outputval)
      
      timings = df_scf_results["Timings"]
      timings.run_name = "run_name_test_blah"
      timings.run_time = run_time
      name = "water_timings_$(timings.options[JCTC.contraction_mode])_$(timings.non_timing_data[JCTC.contraction_algorithm])"
      name = replace(name, " "=> "_")
      println("saving to $(name)")
      save_jc_timings_to_hdf5(timings, joinpath(output_path, "$(name).h5"))
      exit()
      GC.gc(true)
      CUDA.reclaim()
      CUDA.synchronize()
      println("finished warm up")
      
      # cpu_scf_results, cpu_properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted_cpu.json"), output=outputval)
    # end
    # df_scf_results, density_fitted_properties = full_rhf(joinpath(@__DIR__, "../example_inputs/density_fitting/water_density_fitted_gpu.json"), output=outputval)



    for i in 1:1
     
      # sleep(.1)
      full_rhf(denity_fitted_input_file, output=outputval)
      # scf_results, properties = full_rhf(input_file, output=outputval)

      # energy_diff = abs(scf_results["Energy"] - df_scf_results["Energy"])
      # println("Energy difference: $energy_diff")
    end
    # DF_time = @elapsed begin @time begin 
    #   df_scf_results, density_fitted_properties = full_rhf(denity_fitted_input_file)
    # end end
    # RHF_time = @elapsed begin @time begin 
    #   scf_results, properties = full_rhf(input_file)      
    # end end

    

    # println("done iwth runs")
    # flush(stdout)

    # println("---------------------------------------------")
    # println("RHF iteration Times (seconds):")
    # print_iteration_times(scf_results["Timings"])
    
    # println("DF-RHF iteration Times (seconds):")
    # print_iteration_times(df_scf_results["Timings"])

  
    # println("---------------------------------------------")

    # println("RHF energy   : $(scf_results["Energy"]), time: $(RHF_time) seconds")
    # println("DF-RHF energy: $(df_scf_results["Energy"]) time: $(DF_time) seconds")

    # Test.@test scf_results["Energy"] ≈ df_scf_results["Energy"] atol=.00015 #15 micro hartree tolerance
    # println("Test run successfully!")
  # catch e
  #   println("check_density_fitted_method_matches_RHF Failed with exception:\n") 
  #   display(e) 
  #   flush(stdout)
  #   exit()
  # end 
  # JuliaChem.finalize()

end

function main()
  println("Running Density Fitting Tests")
  println(BLAS.get_config())
  println("number of gpus =  $(length(CUDA.devices()))")
  JuliaChem.initialize() 
  println("initialized JuliaChem"); flush(stdout)

  n_threads = Threads.nthreads()

  comm_rank = MPI.Comm_rank(MPI.COMM_WORLD)
  # println("starting JC on rank $comm_rank with $n_threads threads")
  # if comm_rank %2 == 0
  #   ThreadPinning.pinthreads(0:n_threads-1)
  # else
  #   ThreadPinning.pinthreads(n_threads:(n_threads*2)-1)
  # end
  # ThreadPinning.pinthreads(0:n_threads-1)
  BLAS.set_num_threads(n_threads)
  # check_density_fitted_method_matches_RHF(ARGS[1], ARGS[2])

  # df_path = ARGS[1]
  # rhf_path = ARGS[2]

  # df_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/C20H42_dfGPU.json")
  # rhf_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/C20H42_df.json")

  # df_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/C40H82_df.json")
  # rhf_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/C40H82.json")

  # df_path = "/home/jackson/source/JuliaChem.jl/example_inputs/S22_3/6-31+G_d/ammonia_trimer_df.json"
  # rhf_path = "/home/jackson/source/JuliaChem.jl/example_inputs/S22_3/6-31+G_d/benzene_2_water.json"

  # MP2_Num = "03"
  # df_path = joinpath(@__DIR__,  "../example_inputs/density_fitting/$(MP2_Num)_MP2_df.json")
  # rhf_path =  joinpath(@__DIR__, "../example_inputs/density_fitting/$(MP2_Num)_MP2.json")

  # check_density_fitted_method_matches_RHF(df_path, rhf_path, true)


  df_rhf_path = joinpath(@__DIR__,  "/home/jackson/source/JuliaChem.jl/example_inputs/gly/df_gpu/gly")
  rhf_path = joinpath(@__DIR__,  "/home/jackson/source/JuliaChem.jl/example_inputs/gly/df/gly")

  output_path = "/home/jackson/source/JuliaChem.jl/testoutputs/"

  start_index = 1
  end_index = 18
  for i in start_index:end_index
    println("Running polyglycine-$i")
  
      for j in [1]
        # try  
          for k in 1:2 #run each input twice for each JC_K_RECT_N_BLOCKS

          df_gly_path = df_rhf_path * string(i) * ".json"
          rhf_gly_path = rhf_path * string(i) * ".json"


          # df_gly_path = "/home/jackson/source/benchmark_JC/JuliaChem-Benchmarks/DF-RHF-Benchmark/S22_3/cc-pvdz/ammonia_trimer.json"
          # rhf_gly_path =  "/home/jackson/source/benchmark_JC/JuliaChem-Benchmarks/DF-RHF-Benchmark/S22_3/cc-pvdz/ammonia_trimer.json"

          check_density_fitted_method_matches_RHF(df_gly_path, rhf_gly_path, output_path, i == start_index && j == 1)
          display(CUDA.pool_status())
          GC.gc(true)
          CUDA.reclaim()
          CUDA.synchronize()
          display(CUDA.pool_status())
          GC.gc(true)
          CUDA.reclaim()
          CUDA.synchronize()
          display(CUDA.pool_status())
        end
        # catch 
        #   println("Failed on polyglycine-$i run $JC_K_RECT_N_BLOCKS")
        # end
      end
   
  end
end

main()