using Base.Threads
using LinearAlgebra
using HDF5
using PrettyTables
using Printf
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared

using JuliaChem.Shared.JCTC

const do_continue_print = false 
const print_eri = false 
const get_next_index_tag = 1111

function rhf_energy(mol::Molecule, basis_sets::CalculationBasisSets,
  scf_flags::Union{Dict{String,Any},Dict{Any,Any},Dict{String,String}}; output)

  # todo move all of these options to scf_options 
  # todo move this all to a function 
  debug::Bool = haskey(scf_flags, "debug") ? scf_flags["debug"] : false
  
  ndiis::Int = haskey(scf_flags, "ndiis") ? scf_flags["ndiis"] : 10
  dele::Float64 = haskey(scf_flags, "dele") ? scf_flags["dele"] : 1E-6
  rmsd::Float64 = haskey(scf_flags, "rmsd") ? scf_flags["rmsd"] : 1E-6
  fdiff::Bool = haskey(scf_flags, "fdiff") ? scf_flags["fdiff"] : false

  scf_options = create_scf_options(scf_flags)

  if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    print_scf_options(scf_options)
  end
  
  check_auxillary_basis_is_provided(scf_options.density_fitting, basis_sets)

  jc_timing = create_jctiming() :: JCTiming

  set_threads_and_ranks!(jc_timing, Threads.nthreads(), MPI.Comm_size(MPI.COMM_WORLD))

  set_user_scf_options_data!(jc_timing, scf_options)

  kernel_result = rhf_kernel(mol, basis_sets; output=output, debug=debug,
    ndiis=ndiis, dele=dele, rmsd=rmsd,
    fdiff=fdiff, scf_options=scf_options, jc_timing)

  set_scf_options_data!(jc_timing, scf_options) # scf_options can be overwritten by the various algorithms
  return kernel_result
end


"""
	 rhf_kernel(FLAGS::RHF_Flags, basis_sets::CalculationBasisSets, read_in::Dict{String,Any},
       type::T)
Summary
======
Perform the core RHF SCF algorithm.

Arguments
======
FLAGS = Input flags

basis = Generated basis set

read_in = file required to read in from input file

type = Precision of variables in calculation

method = the method to calculate the SCF E.G. RHF (Restricted Hatree Fock) or DFRIHF (Density fitted RHF)
"""
function rhf_kernel(mol::Molecule, 
  basis_sets::CalculationBasisSets; 
  output::Int64, debug::Bool, ndiis::Int, 
  dele::Float64, rmsd::Float64, fdiff::Bool, scf_options::SCFOptions, jc_timing::JCTiming)
  
  basis = basis_sets.primary
  comm=MPI.COMM_WORLD
  calculation_status = Dict([])

  #== read in some variables from scf input ==#
  debug_output = debug ? h5open("debug.h5","w") : nothing

  #== compute nuclear repulsion energy ==# 
  E_nuc = compute_enuc(mol)
  
  jeri_oei_engine = JERI.OEIEngine(mol.mol_cxx, 
    basis.basis_cxx, 0) 
  
  #== compute one-electron integrals and Hamiltonian ==#
  S = zeros(Float64, (basis.norb, basis.norb))
  compute_overlap(S, basis, jeri_oei_engine)
 
  #for i in 1:basis.norb, j in 1:i
  #  println("OVR($i,$j): ", S[i,j])
  #end

  T = zeros(Float64, (basis.norb, basis.norb))  
  V = zeros(Float64, (basis.norb, basis.norb))
  H = zeros(Float64, (basis.norb, basis.norb))

  T_time = @elapsed compute_ke(T, basis, jeri_oei_engine)
  V_time = @elapsed compute_nah(V, mol, basis, jeri_oei_engine)

  H_time = @elapsed H .= T .+ V

  jc_timing.timings[JCTC.H_time] = T_time + V_time + H_time
 
  #== build the initial matrices with guesss ==#
  if scf_options.guess == Guess.sad
    guess_matrix = sad_guess(mol, basis)
    D = guess_matrix
    F = zeros(size(H))
  elseif scf_options.guess == Guess.hcore || scf_options.guess == Guess.density_fitting
    guess_matrix = deepcopy(H)
    F = guess_matrix
    D = zeros(size(H))
  else
    error("Guess $(scf_options.guess) not a valid guess mode")
  end

  if debug && MPI.Comm_rank(comm) == 0
    h5write("debug.h5","RHF/Iteration-None/E_nuc", E_nuc)
    h5write("debug.h5","RHF/Iteration-None/S", S)
    h5write("debug.h5","RHF/Iteration-None/T", T)
    h5write("debug.h5","RHF/Iteration-None/V", V)
    h5write("debug.h5","RHF/Iteration-None/H", H)
    h5write("debug.h5","RHF/Iteration-None/Guess", guess_matrix)
  end

  #== build the initial matrices ==#
  F_eval = Vector{Float64}(undef,basis.norb)
  F_evec = zeros(size(F))

  W = zeros(size(F))
  
  C = zeros(size(F))

  #== allocate workspace matrices ==#
  workspace_a = zeros(size(F))
  workspace_b = zeros(size(F))
  workspace_c = [ similar(F) ]

  #== check for linear dependencies ==#
  minimum_eval_threshold = 1.0E-6 
  LinearAlgebra.BLAS.blascopy!(length(S), S, 1, workspace_b, 1)
  #workspace_b .= S
  S_eval_diag, workspace_a[:,:] = eigen!(LinearAlgebra.Hermitian(workspace_b))

  #fill!(workspace_b, 0.0)
  LinearAlgebra.BLAS.scal!(length(workspace_b), 0.0, workspace_b, 1) 
  for i in 1:basis.norb
    workspace_b[i,i] = S_eval_diag[i]
  end

  S_good_evals = findall(eval -> eval >= minimum_eval_threshold, S_eval_diag)  
  #ortho = similar(F)
  #LinearAlgebra.BLAS.gemm!('N', 'T', 1.0, 
  #  workspace_b, workspace_a, 0.0, ortho)
  #LinearAlgebra.BLAS.gemm!('N', 'N', 1.0, workspace_a, ortho, 0.0, ortho) 
 
  #== build the orthogonalization matrix ==#
  @views ortho = workspace_a[1:end,S_good_evals]*
    (LinearAlgebra.Diagonal(workspace_b[S_good_evals,S_good_evals])^-0.5)*
    transpose(workspace_a[1:end, S_good_evals])
  
  if debug && MPI.Comm_rank(comm) == 0
    h5write("debug.h5","RHF/Iteration-None/X", ortho)
  end

  if MPI.Comm_rank(comm) == 0 && output >= 2
    println("----------------------------------------          ")
    println("       Starting RHF iterations...                 ")
    println("----------------------------------------          ")
    println(" ")
    println("Iter        Energy                ΔE                Drms                time")
  end

  E_elec = 0.0
  F_eval = zeros(size(F)[1])
  if scf_options.guess == Guess.hcore || scf_options.guess == Guess.density_fitting
    E_elec, F_eval[:] = iteration(F, D, C, H, F_eval, F_evec, workspace_a, 
      workspace_b, ortho, basis_sets, 0, debug)
  end
  
  F_old = deepcopy(F)
  
  E = 0.0 
  E_old = E

  #if MPI.Comm_rank(comm) == 0 && output >= 2
  #  @printf("0     %.10f\n", E)
  #end

  #=============================#
  #== start scf cycles: #7-10 ==#
  #=============================#
  F, D, W, C, E, converged = scf_cycles(F, D, W, C, E, H, ortho, S, 
    F_eval, F_evec, F_old, workspace_a, workspace_b, workspace_c,
    E_nuc, E_elec, E_old, basis_sets; 
    output=output, debug=debug, ndiis=ndiis, dele=dele,
    rmsd=rmsd, fdiff=fdiff, scf_options = scf_options, jc_timing)

  if !converged
    if MPI.Comm_rank(comm) == 0 && output >= 1
      println(" ")
      println("----------------------------------------")
      println(" The SCF calculation did not converge.  ")
      println("      Restart data is being output.     ")
      println("----------------------------------------")
      println(" ")
    end

    calculation_fail = Dict(
    "success" => false,
    "error" => Dict(
      "error_type" => "convergence_error",
      "error_message" => " SCF calculation did not converge within $(scf_options.max_iterations)
        iterations. "
      )
      
    )

    merge!(calculation_status, calculation_fail)

  else
    if MPI.Comm_rank(comm) == 0 && output >= 1 
      println(" ")
      println("----------------------------------------")
      println("   The SCF calculation has converged!   ")
      println("----------------------------------------")
      #println("Total SCF Energy: ",E," h")
      @printf("Total SCF Energy: %.10f h\n",E)
      println(" ")

      calculation_success = Dict(
      "return_result" => E,
      "success" => true,
      "properties" => Dict(
        "return_energy" => E,
        "nuclear_repulsion_energy" => E_nuc,
        #"scf_iterations" => iter,
        "scf_total_energy" => E
        )
      )

      merge!(calculation_status, calculation_success)
    end
  end

  if debug close(debug_output) end

  #todo(Jackson) this should be a struct
  scf = Dict("Fock" => F,                                                       
             "Density" => D,                                                    
             "Energy-Weighted Density" => W,                                                    
             "MO Coeff" => C,                                                   
             "Overlap" => S,                                                   
             "Energy" => E,                                                     
             "Converged?" => converged,
             "Timings"=> jc_timing                                              
            )      
  
  return scf 
end

function scf_cycles(F::Matrix{Float64}, D::Matrix{Float64}, 
  W::Matrix{Float64}, C::Matrix{Float64},
  E::Float64, H::Matrix{Float64}, ortho::Matrix{Float64}, 
  S::Matrix{Float64}, F_eval::Vector{Float64}, 
  F_evec::Matrix{Float64},  F_old::Matrix{Float64},
  workspace_a::Matrix{Float64}, workspace_b::Matrix{Float64}, 
  workspace_c::Vector{Matrix{Float64}}, E_nuc::Float64, E_elec::Float64, 
  E_old::Float64, basis_sets::CalculationBasisSets;
  output::Int64, debug::Bool, ndiis::Int, 
  dele::Float64, rmsd::Float64, fdiff::Bool, scf_options::SCFOptions, jc_timing::JCTiming)
  basis = basis_sets.primary
  #== read in some more variables from scf flags input ==#
  nsh = length(basis)
  nindices = (muladd(nsh,nsh,nsh)*(muladd(nsh,nsh,nsh) + 2)) >> 3

  #== build DIIS arrays ==#
  F_array = fill(similar(F), ndiis)

  e_array = fill(similar(F), ndiis)
  e_array_old = fill(similar(F), max(0,ndiis-1))
  
  F_array_old = fill(similar(F), max(0,ndiis-1))

  #FD = similar(F)
  FDS = similar(F)
  #SDF = similar(F)
  
  #== build arrays needed for post-fock build iteration calculations ==#
  #F_temp = similar(F)
  ΔF = similar(F) 
  F_cumul = zeros(size(F)) 
 
  D_old = zeros(size(F))
  ΔD = deepcopy(D) 
  D_input = similar(F)

  #== build matrix of Cauchy-Schwarz upper bounds ==# 
  schwarz_bounds = zeros(Float64,(nsh,nsh)) 
  compute_schwarz_bounds(schwarz_bounds, basis, nsh)

  Dsh = similar(schwarz_bounds)
  
  #== build eri batch arrays ==#
  #eri_sizes::Vector{Int64} = load("tei_batch.jld",
  #  "Sizes/$quartet_batch_num_old")
  #length_eri_sizes::Int64 = length(eri_sizes)

  #@views eri_starts::Vector{Int64} = [1, [ sum(eri_sizes[1:i])+1 for i in 1:(length_eri_sizes-1)]... ]

  #eri_batch::Vector{Float64} = load("tei_batch.jld",
  #  "Integrals/$quartet_batch_num_old")

  #eri_sizes = []
  #eri_starts = []
  #eri_batch = []

  #== execute convergence procedure ==#
  E, scf_converged = scf_cycles_kernel(F, D, W, C, E, H, ortho, S, E_nuc,
    E_elec, E_old, basis_sets, F_array, e_array, e_array_old,
    F_array_old, F_eval, F_evec, F_old, workspace_a, 
    workspace_b, workspace_c, ΔF, F_cumul, 
    D_old, ΔD, D_input, FDS, 
    schwarz_bounds, Dsh; 
    output=output, debug=debug, ndiis=ndiis, dele=dele, 
    rmsd=rmsd, fdiff=fdiff, scf_options, jc_timing)
  #== we are done! ==#
  if debug
    h5write("debug.h5","RHF/Iteration-Final/F", F)
    h5write("debug.h5","RHF/Iteration-Final/D", D)
    h5write("debug.h5","RHF/Iteration-Final/C", C)
    h5write("debug.h5","RHF/Iteration-Final/E", E)
    h5write("debug.h5","RHF/Iteration-Final/converged", scf_converged)
  end
  return F, D, W, C, E, scf_converged
end

function scf_cycles_kernel(F::Matrix{Float64}, D::Matrix{Float64},
  W::Matrix{Float64}, C::Matrix{Float64}, E::Float64, H::Matrix{Float64}, 
  ortho::Matrix{Float64}, S::Matrix{Float64}, E_nuc::Float64, 
  E_elec::Float64, E_old::Float64, basis_sets::CalculationBasisSets,
  F_array::Vector{Matrix{Float64}}, 
  e_array::Vector{Matrix{Float64}}, e_array_old::Vector{Matrix{Float64}},
  F_array_old::Vector{Matrix{Float64}}, 
  F_eval::Vector{Float64}, F_evec::Matrix{Float64}, 
  F_old::Matrix{Float64}, workspace_a::Matrix{Float64}, 
  workspace_b::Matrix{Float64}, workspace_c::Vector{Matrix{Float64}}, 
  ΔF::Matrix{Float64},
  F_cumul::Matrix{Float64}, D_old::Matrix{Float64}, 
  ΔD::Matrix{Float64}, D_input::Matrix{Float64},  
  FDS::Matrix{Float64}, 
  schwarz_bounds::Matrix{Float64}, Dsh::Matrix{Float64}; 
  output, debug, ndiis, dele, rmsd, fdiff, scf_options:: SCFOptions, jc_timing::JCTiming)
  basis = basis_sets.primary
  auxiliary_basis = basis_sets.auxillary
 
  scf_converged::Bool = true
  iter_start = time()
  #== initialize a few more variables ==#
  comm=MPI.COMM_WORLD

  B_dim = 1
  D_rms = 1.0
  ΔE = 1.0 
  cutoff = fdiff ? 5E-11 : 1E-10

  #length_eri_sizes = length(eri_sizes)

  #=================================#
  #== now we start scf iterations ==#
  #=================================#
  iter = 1
  iter_converged = false
  nthreads = Threads.nthreads()
  
  max_am = max_ang_mom(basis) 
  eri_quartet_batch_thread = [ Vector{Float64}(undef,
    eri_quartet_batch_size(max_am)) 
    for thread in 1:nthreads ]
 
  F_thread = [ zeros(size(F)) for thread in 1:nthreads ]

  do_density_fitting = scf_options.density_fitting || scf_options.guess == Guess.density_fitting

  jeri_engine_thread_df = do_density_fitting ? [JERI.DFRHFTEIEngine(basis.basis_cxx, auxiliary_basis.basis_cxx, basis.shpdata_cxx, auxiliary_basis.shpdata_cxx) for thread in 1:nthreads ] : []
  jeri_engine_thread = [JERI.RHFTEIEngine(basis.basis_cxx, basis.shpdata_cxx)  for thread in 1:nthreads ]
  gpu_data = SCFGPUDataNone()
  if scf_options.contraction_mode == "GPU" || scf_options.contraction_mode == "denseGPU"

    # gpu_data = get_default_gpu_data_cuda() #CUDA GPU
    #check if AMD is available
    if CUDA_GPU_enabled()
      gpu_data = get_default_gpu_data_cuda() #CUDA GPU
      println("using CUDA GPU")
    elseif AMD_GPU_enabled()
      gpu_data = get_default_gpu_data_AMD() #AMD GPU
      println("using AMD GPU")
    end
  end

  scf_data = SCFData(gpu_data)

  density_fitting_converged = false

  set_basis_info!(jc_timing, basis, auxiliary_basis)
  
  while !iter_converged
    just_finished_df_iters = false
    iteration_start = time()
    flush(stdout)
    #== reset eri arrays ==#
    #if quartet_batch_num_old != 1 && iter != 1
    #  resize!(eri_sizes,length_eri_sizes)
    #  resize!(eri_starts,length_eri_sizes)

    #  eri_sizes[:] = load("tei_batch.jld",
  #      "Sizes/$quartet_batch_num_old")

    #  @views eri_starts[:] = [1, [ sum(eri_sizes[1:i])+1 for i in 1:(length_eri_sizes-1)]... ]
      #eri_starts[:] = load("tei_batch.jld",
      #  "Starts/$quartet_batch_num_old")
      #@views eri_starts[:] = eri_starts[:] .- (eri_starts[1] - 1)

    #  resize!(eri_batch,sum(eri_sizes))
    #  eri_batch[:] = load("tei_batch.jld","Integrals/$quartet_batch_num_old")
    #end

    #== determine input D and F ==#
    if fdiff
      LinearAlgebra.BLAS.blascopy!(length(ΔD), ΔD, 1, D_input, 1)
      LinearAlgebra.BLAS.blascopy!(length(ΔF), ΔF, 1, workspace_b, 1)
      #D_input .= fdiff ? ΔD : D
      #workspace_b .= fdiff ? ΔF : F
    else
      LinearAlgebra.BLAS.blascopy!(length(D), D, 1, D_input, 1)
      LinearAlgebra.BLAS.blascopy!(length(F), F, 1, workspace_b, 1)
      #D_input .= fdiff ? ΔD : D
      #workspace_b .= fdiff ? ΔF : F
    end
     
    #== build new Fock matrix ==#
    if !do_density_fitting
      #== compress D into shells in Dsh ==#
      for ish in 1:length(basis), jsh in 1:ish
        ipos = basis[ish].pos
        ibas = basis[ish].nbas

        jpos = basis[jsh].pos
        jbas = basis[jsh].nbas
        
        max_value = 0.0
        for i in ipos:(ipos+ibas-1), j in jpos:(jpos+jbas-1) 
          max_value = max(max_value, Base.abs_float(D_input[i,j]))
        end
        Dsh[ish, jsh] = max_value
        Dsh[jsh, ish] = Dsh[ish, jsh] 
      end

      fock_build_start_time =  time()
      rfh_fock_build(workspace_a, workspace_b, F, 
      F_thread, D, 
      H, basis_sets, 
      schwarz_bounds, Dsh,
      eri_quartet_batch_thread, 
      jeri_engine_thread, iter,
      cutoff, debug, scf_options.load, fdiff, ΔF, F_cumul)     
      fock_build_end_time = time()
      jc_timing.timings[JCTiming_key(JCTC.fock_time,iter)] = fock_build_end_time - fock_build_start_time 
    else
      MPI.Bcast!(C, 0, comm)
      fock_build_time = @elapsed F = df_rhf_fock_build!(scf_data, jeri_engine_thread_df, jeri_engine_thread, basis_sets, C, iter, scf_options, H, jc_timing)
      jc_timing.timings[JCTiming_key(JCTC.fock_time,iter)] = fock_build_time
    end
    
    if debug && MPI.Comm_rank(comm) == 0
      h5write("debug.h5","RHF/Iteration-$iter/F/Total", F)
    end

    #== do DIIS ==#
    if ndiis > 0
      BLAS.symm!('L', 'U', 1.0, F, D, 0.0, workspace_a)
      BLAS.gemm!('N', 'N', 1.0, workspace_a, S, 0.0, FDS)
      
      transpose!(workspace_b, FDS)
    
      #== compute error vector ==# 
      LinearAlgebra.BLAS.blascopy!(length(FDS), FDS, 1, 
        workspace_a, 1) 
      axpy!(-1.0, workspace_b, workspace_a)  

      e_array_old = view(e_array,1:(ndiis-1))                                   
      workspace_c[1] = deepcopy(workspace_a)
      e_array = vcat(workspace_c, e_array_old)                                                                          
      F_array_old = view(F_array,1:(ndiis-1))                                   
      workspace_c[1] = deepcopy(F)
      F_array = vcat(workspace_c, F_array_old)              
      
      if iter > 1
        B_dim += 1
        B_dim = min(B_dim,ndiis)
        try
          DIIS(F, e_array, F_array, B_dim)
        catch
          println("Faulty DIIS!")
          B_dim = 2
          #DIIS(F, e_array, F_array, B_dim)
        end
      end
    end

    #== dynamic damping of Fock matrix ==#
    x = ΔE >= 1.0 ? 1.0/log(50,50*ΔE) : 1.0 
    LinearAlgebra.BLAS.axpby!(1.0-x, F_old, x, F)

    LinearAlgebra.BLAS.blascopy!(length(F), F, 1, 
      F_old, 1) 
  
    #== obtain new F,D,C matrices ==#
    LinearAlgebra.BLAS.blascopy!(length(D), D, 1, 
      D_old, 1) 
      E_elec, F_eval[:] = iteration(F, D, C, H, F_eval, F_evec, workspace_a,
      workspace_b, ortho, basis_sets, iter, debug)

    #== check for convergence ==#
    #LinearAlgebra.BLAS.blascopy!(length(FDS), FDS, 1, 
    #  workspace_a, 1) 
    #axpy!(-1.0, workspace_b, workspace_a)  

    ΔD .= D .- D_old
    D_rms = √(LinearAlgebra.BLAS.dot(length(ΔD), ΔD, 1, ΔD, 1))

    E = E_elec + E_nuc
    ΔE = E - E_old

    if do_density_fitting && !density_fitting_converged 
      density_fitting_converged = 
        Base.abs_float(ΔE) <= scf_options.df_energy_convergence &&
        D_rms <= scf_options.df_density_convergence

      # max iter for density fitting should be a parameter
      if density_fitting_converged || iter > scf_options.df_max_iterations
        xyK = nothing
        xiK = nothing
        two_electron_fock_component = nothing        
        do_density_fitting = false
        jc_timing.non_timing_data[JCTC.DF_iteration_range_start] = string(2)
        jc_timing.non_timing_data[JCTC.DF_iteration_range_end] = string(iter)
        
        if scf_options.guess == Guess.density_fitting # density fitting guess done proceed to RHF 
          jeri_engine_thread = [JERI.RHFTEIEngine(basis.basis_cxx, basis.shpdata_cxx) for thread in 1:nthreads ]
          just_finished_df_iters = true
        else # Density Fitting RHF done 
          break 
        end
      end
    else
      iter_converged = Base.abs_float(ΔE) <= dele && D_rms <= rmsd
    end

    iteration_time = time() - iteration_start

    jc_timing.timings[JCTiming_key(JCTC.iteration_time,iter)] = iteration_time
    if MPI.Comm_rank(comm) == 0 && output >= 2
      #println(iter,"     ", E,"     ", ΔE,"     ", D_rms)
      @printf("%d      %.10f      %.10f      %.10f      %.10f \n", iter, E, ΔE, D_rms, iteration_time)
    end

    if just_finished_df_iters && MPI.Comm_rank(comm) == 0 && output >= 2
      println("------------ done with density fitted iterations --------------") 
      just_finished_df_iters = false
    end 
    
    if maximum_iterations_exceeded(iter, scf_options)
      scf_converged = false
      break
    end
    iter += 1
    scf_data.scf_iteration = iter  
    #== if not converged, replace old D and E values for next iteration ==#
    E_old = E
  end

  

  #== build energy-weighted density matrix ==#
  #fill!(W, 0.0)
  LinearAlgebra.BLAS.scal!(length(W), 0.0, W, 1) 
  
  nocc = basis.nels >> 1
  for i in 1:basis.norb, j in 1:basis.norb
    for iocc in 1:nocc
      W[i,j] += 2.0 * F_eval[iocc] * C[i, iocc] * C[j, iocc]
    end
  end

  set_converge_properties!(jc_timing, scf_converged, iter, E)


  return E, scf_converged
end



function maximum_iterations_exceeded(iter, scf_options) :: Bool
  if scf_options.density_fitting 
    return iter >= scf_options.df_max_iterations  
  elseif scf_options.guess == Guess.density_fitting ## density fitting is guess and in df mode
    return iter >= scf_options.max_iterations + scf_options.df_max_iterations
  else  # no density fitting
    return iter >= scf_options.max_iterations
  end
end

#== Restricted Hartree-Fock, Fock build step ==#
function rfh_fock_build(workspace_a, workspace_b, F, 
  F_thread, D, 
  H, basis_sets, 
  schwarz_bounds, Dsh,
  eri_quartet_batch_thread, 
  jeri_engine_thread, iter,
  cutoff, debug, load, fdiff, ΔF, F_cumul)

  workspace_a .= run_rfh_fock_build_kernel(F, 
  F_thread, D, 
  H, basis_sets, 
  schwarz_bounds, Dsh,
  eri_quartet_batch_thread, 
  jeri_engine_thread, iter,
  cutoff, debug, load)

  workspace_b .= MPI.Allreduce(workspace_a,MPI.SUM,MPI.COMM_WORLD)
  MPI.Barrier(MPI.COMM_WORLD)

  if debug && MPI.Comm_rank(MPI.COMM_WORLD) == 0
    h5write("debug.h5","RHF/Iteration-$iter/F/Skeleton", workspace_b)
  end

  if fdiff 
    ΔF .= workspace_b
    F_cumul .+= ΔF
    F .= F_cumul .+ H
  else
    LinearAlgebra.BLAS.axpy!(1.0, H, workspace_b) 
    LinearAlgebra.BLAS.blascopy!(length(workspace_b), workspace_b, 1, 
      F, 1) 

    #F .= workspace_b .+ H
  end
  return F
end



#=
"""
	 run_rfh_fock_build_kernel(F::Array{Float64}, D::Array{Float64}, tei::Array{Float64}, H::Array{Float64})
Summary
======
Perform Fock build step.

Arguments
======
F = Current iteration's Fock Matrix

D = Current iteration's Density Matrix

tei = Two-electron integral array

H = One-electron Hamiltonian Matrix
"""
=#

@inline function run_rfh_fock_build_kernel(F::Matrix{Float64}, 
  F_thread::Vector{Matrix{Float64}}, D::Matrix{Float64}, 
  H::Matrix{Float64}, basis_sets::CalculationBasisSets, 
  schwarz_bounds::Matrix{Float64}, Dsh::Matrix{Float64},
  eri_quartet_batch_thread::Vector{Vector{Float64}}, 
  jeri_engine_thread, iter::Int64,
  cutoff::Float64, debug::Bool, load::String)
  basis = basis_sets.primary

  comm = MPI.COMM_WORLD
  
  #fill!(F,zero(Float64))
  LinearAlgebra.BLAS.scal!(length(F), 0.0, F, 1) 
  #fill!.(F_thread,zero(Float64)) 
  for ithread_fock in F_thread
    LinearAlgebra.BLAS.scal!(length(ithread_fock), 0.0, ithread_fock, 1) 
  end

  nsh = length(basis)
  nindices = get_n_shell_indicies(nsh)
  n_threads = Threads.nthreads()
  batches_per_thread = nsh 
  batch_size = ceil(Int,nindices/(MPI.Comm_size(comm)* n_threads*batches_per_thread))

  #== sequential method is here for debugging, not reccomended for actual calculations ==#
  if load == "sequential"
    #== set up initial indices ==#                                              
    stride = 1                                  
    top_index = nindices
                                                                                
    #== execute kernel of calculation ==#                                       
    for ijkl in top_index:-stride:1                     
        thread = Threads.threadid()                                             
      
        eri_quartet_batch_priv = eri_quartet_batch_thread[thread]               
        jeri_tei_engine_priv = jeri_engine_thread[thread]                       
        F_priv = F_thread[thread]         

        fock_build_thread_kernel(F_priv, D,                                 
          H, basis_sets, eri_quartet_batch_priv,                                 
          ijkl, jeri_tei_engine_priv,                                       
          schwarz_bounds, Dsh,                                              
          cutoff, debug)                
                                                                               
    end     
    #== reduce into Fock matrix ==#    
    axpy!(1.0, F_thread[1], F) 
  #== use static task distribution for multirank runs if selected... ==#
  elseif load == "static" || (MPI.Comm_size(comm) == 1 && n_threads == 1)
    #== set up initial indices ==#                                              
    stride = MPI.Comm_size(comm)*batch_size                                       
    top_index = nindices - (MPI.Comm_rank(comm)*batch_size)
                                                                                
    #== execute kernel of calculation ==#                                       
    @sync for ijkl_index in top_index:-stride:1                     
      Threads.@spawn begin                                                      
        thread = Threads.threadid()                                             
      
        eri_quartet_batch_priv = eri_quartet_batch_thread[thread]               
        jeri_tei_engine_priv = jeri_engine_thread[thread]                       
        F_priv = F_thread[thread]                                               

        for ijkl in ijkl_index:-1:(max(1,ijkl_index-batch_size+1))   
          fock_build_thread_kernel(F_priv, D,                                 
            H, basis_sets, eri_quartet_batch_priv,                                 
            ijkl, jeri_tei_engine_priv,                                       
            schwarz_bounds, Dsh,                                              
            cutoff, debug)                                                    
        end
      end                                                                       
    end      

    #== reduce into Fock matrix ==#
    for (thread,ithread_fock) in enumerate(F_thread)
      if debug && MPI.Comm_rank(comm) == 0
        h5write("debug.h5","RHF/Iteration-$iter/F/Thread-$thread", ithread_fock)
      end
 
      axpy!(1.0, ithread_fock, F) 
    end
 
  #== ..else use dynamic task distribution ==# 
  elseif load == "dynamic"
    do_dynamic_load_scf(F, nindices, batch_size, D, H, eri_quartet_batch_thread, jeri_engine_thread, 
    F_thread, basis_sets, schwarz_bounds, Dsh, cutoff, debug, iter)
  end
  MPI.Barrier(comm)

  for iorb in 1:basis.norb, jorb in 1:iorb
    if iorb != jorb
      F[iorb,jorb] /= 2.0
      F[jorb,iorb] = F[iorb,jorb]
    end
  end

  return F
end

@inline function do_dynamic_load_scf(F, nindices, batch_size, D, H, eri_quartet_batch_thread, jeri_engine_thread, 
    F_thread, basis_sets, schwarz_bounds, Dsh, cutoff, debug, iter)

  comm = MPI.COMM_WORLD
  task = [ nindices ]
  rank = MPI.Comm_rank(comm)
  n_ranks = MPI.Comm_size(comm)
  n_threads = Threads.nthreads()
  #== create needed mutex ==#
  mutex_mpi_worker = Base.Threads.ReentrantLock()
  # mutex_mpi_coordinator = Base.Threads.ReentrantLock()
  @sync for thread_number in 1:n_threads
    Threads.@spawn begin       
      threadid = Threads.threadid()
      if rank == 0 && thread_number == 1 && n_ranks > 1

        #== coordinator rank ==#
        #== hand out quartets to workers dynamically ==#

        send_indicies_dynamic(task, batch_size, mutex_mpi_worker, thread_number)
                  
      # elseif rank > 0 # todo remove this line after DF-RHF paper is published worker ranks perform actual computations on quartets old mode where the 0 rank does no work uncomment and comment the else below. this to use the old method should only be used for comparison
      else # worker ranks perform actual computations on quartets new mode 
        do_dynamic_worker(task, thread_number, batch_size, F_thread, D,
        H, basis_sets, eri_quartet_batch_thread,
        jeri_engine_thread,
        schwarz_bounds, Dsh,
        cutoff, debug, mutex_mpi_worker)
      end 
    end  # end threads
  end # end sync
  #== reduce into Fock matrix ==#
  
  cleanup_messages_scf(n_ranks)
  MPI.Barrier(comm)

  for ithread_fock in F_thread 
    axpy!(1.0, ithread_fock, F)
  end
end

function cleanup_messages_scf(n_ranks)
  comm = MPI.COMM_WORLD
  if n_ranks > 1 
    ismessage = true 
    while ismessage
      ismessage, status = MPI.Iprobe(comm, MPI.Status; source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
      if ismessage
        recv_mesg = [0]
        if status.source == 0
          recv_mesg = [0]
        else  
          recv_mesg = [0,0]
          data, recv_status = MPI.Recv!(recv_mesg, comm, MPI.Status; source=status.source, tag=status.tag)
        end     
      end
    end 
  end
end

function do_dynamic_worker(task, threadid, batch_size, F_thread, D,
  H, basis_sets, eri_quartet_batch_thread,
  jeri_engine_thread,
  schwarz_bounds, Dsh,
  cutoff, debug, mutex_mpi_worker)
  
  comm = MPI.COMM_WORLD
  rank = MPI.Comm_rank(comm)

  eri_quartet_batch_priv = eri_quartet_batch_thread[threadid] 
  jeri_tei_engine_priv = jeri_engine_thread[threadid] 
  F_priv = F_thread[threadid] 

  #== execute kernel ==# 
  ijkl_index = get_next_index_scf(task, batch_size, mutex_mpi_worker, threadid)
  while ijkl_index >= 1 
    for ijkl in ijkl_index:-1:(max(1,ijkl_index-batch_size+1))
      fock_build_thread_kernel(F_priv, D,
        H, basis_sets, eri_quartet_batch_priv,
        ijkl, jeri_tei_engine_priv,
        schwarz_bounds, Dsh,
        cutoff, debug)
    end
    ijkl_index = get_next_index_scf(task, batch_size, mutex_mpi_worker, threadid)
  end
end

function send_indicies_dynamic(task, batch_size, mutex_mpi_worker, threadid)
  comm = MPI.COMM_WORLD
  while task[1] > 0 
    recv_mesg = [0,0]
    status = MPI.Probe(comm, MPI.Status; source=MPI.ANY_SOURCE,tag=get_next_index_tag) 

    rreq = MPI.Recv!(recv_mesg,comm; source=status.source, tag=get_next_index_tag)  
    index = get_next_index_scf(task, batch_size, mutex_mpi_worker, threadid)
    if index > 0
      sreq = MPI.Send([index], comm; dest=recv_mesg[1], tag=recv_mesg[2])  
    end
  end
  #== hand out ending signals once done ==#

  for rank in 1:(MPI.Comm_size(comm)-1)
    for thread in 1:Threads.nthreads()
      sreq = MPI.Isend([ -1 ], comm; dest=rank, tag=thread) 
    end
  end 
  
end

function get_next_index_scf(task, batch_size ,mutex_mpi_worker, threadid)
  lock(mutex_mpi_worker) do 
    comm = MPI.COMM_WORLD
    next_index = 0
    rank = MPI.Comm_rank(comm)
    if rank == 0
      next_index = task[1]
      task[1] -= batch_size
      return next_index
    else
        send_mesg = [MPI.Comm_rank(comm), threadid]
        MPI.Send(send_mesg, comm; dest=0, tag=get_next_index_tag)
        recv_mesg = [0]
        MPI.Recv!(recv_mesg,comm; source=0, tag=threadid)
        return recv_mesg[1]
    end
  end
end

@inline function fock_build_thread_kernel(F::Matrix{Float64}, D::Matrix{Float64},
  H::Matrix{Float64}, basis_sets::CalculationBasisSets, 
  eri_quartet_batch::Vector{Float64}, 
  ijkl_index::Int64, 
  jeri_tei_engine, schwarz_bounds::Matrix{Float64}, 
  Dsh::Matrix{Float64}, cutoff::Float64, debug::Bool)
  basis = basis_sets.primary

  comm=MPI.COMM_WORLD
  
  #== determine shells==# 
  bra_pair, ket_pair, ish,jsh,ksh,lsh = decompose_shell_index_ijkl(ijkl_index)

  μsh = basis[ish] 
  νsh = basis[jsh] 
  λsh = basis[ksh] 
  σsh = basis[lsh] 
  
  #icls = unsafe_string(μsh.class)
  #jcls = unsafe_string(νsh.class) 
  #kcls = unsafe_string(λsh.class) 
  #lcls = unsafe_string(σsh.class)

  #println("QUARTET($ish, $jsh, $ksh, $lsh) -> ($icls $jcls | $kcls $lcls)")

  #== Cauchy-Schwarz screening ==#
  bound = schwarz_bounds[ish, jsh]*schwarz_bounds[ksh, lsh] 
  
  #println("SCHWARZ($ish, $jsh): $(schwarz_bounds[ish, jsh])")
  
  dijmax = 4.0*Dsh[ish, jsh]
  dklmax = 4.0*Dsh[ksh, lsh]
  
  dikmax = Dsh[ish, ksh]
  dilmax = Dsh[ish, lsh]
  djkmax = Dsh[jsh, ksh]
  djlmax = Dsh[jsh, lsh]
 
  maxden = max(dijmax, dklmax, dikmax, dilmax, djkmax, djlmax)
  bound *= maxden

  #== fock build for significant shell quartets ==# 
  if Base.abs_float(bound) >= 0 
    #= set up some variables =#
    nμ = μsh.nbas
    nν = νsh.nbas
    nλ = λsh.nbas
    nσ = σsh.nbas

    #== compute electron repulsion integrals ==#
    screened = compute_eris(ish, jsh, ksh, lsh, bra_pair, ket_pair, nμ, nν, 
      nλ, nσ, eri_quartet_batch, jeri_tei_engine)
  
    if !screened
      #= axial normalization =#
      axial_normalization_factor(eri_quartet_batch, μsh, νsh, λsh, σsh,
        nμ, nν, nλ, nσ)
 
      #== contract ERIs into Fock matrix ==#
      contract_eris(F, D, eri_quartet_batch, ish, jsh, ksh, lsh,
        μsh, νsh, λsh, σsh, cutoff, debug)
    end
  end
  #if debug println("END TWO-ELECTRON INTEGRALS") end
end

@inline function compute_eris(ish::Int64, jsh::Int64, ksh::Int64, lsh::Int64,
  bra_pair::Int64, ket_pair::Int64, 
  nμ::Int64, nν::Int64,
  nλ::Int64, nσ::Int64,
  eri_quartet_batch::Vector{Float64},
  jeri_tei_engine)

  #= actually compute integrals =#
  return JERI.compute_eri_block(jeri_tei_engine, eri_quartet_batch, 
    ish, jsh, ksh, lsh, bra_pair, ket_pair, nμ*nν, nλ*nσ)
  
  #=
  if am[1] == 3 || am[2] == 3 || am[3] == 3 || am[4] == 3
    for idx in 1:nμ*nν*nλ*nσ 
    #for idx in 1:1296
      eri = eri_quartet_batch[idx]
      println("QUARTET($ish, $jsh, $ksh, $lsh): $eri")
    end
  end
  =#
end

@inline function contract_eris(F_priv::Matrix{Float64}, D::Matrix{Float64},
  eri_batch::Vector{Float64}, ish::Int64, jsh::Int64,
  ksh::Int64, lsh::Int64, 
  μsh::JCModules.Shell, νsh::JCModules.Shell, 
  λsh::JCModules.Shell, σsh::JCModules.Shell,
  cutoff::Float64, debug::Bool)

  #norb = size(D,1)
  
  #ish = μsh.shell_id
  #jsh = νsh.shell_id
  #ksh = λsh.shell_id
  #lsh = σsh.shell_id

  pμ = μsh.pos
  nμ = μsh.nbas

  pν = νsh.pos
  nν = νsh.nbas
  
  pλ = λsh.pos
  nλ = λsh.nbas
  
  pσ = σsh.pos
  nσ = σsh.nbas

  #amμ = μsh.am
  #amν = νsh.am
  #amλ = λsh.am
  #amσ = σsh.am
  #am = [ amμ, amν, amλ, amσ ]

  for μμ::Int64 in pμ:(pμ+nμ-1) 
    νmax = ish == jsh ? μμ : (pν+nν-1)
    for νν::Int64 in pν:νmax
      for λλ::Int64 in pλ:(pλ+nλ-1) 
        σmax = ksh == lsh ? λλ : (pσ+nσ-1)
        for σσ::Int64 in pσ:σmax
          #if debug
            #if do_continue_print print("$μμ, $νν, $λλ, $σσ => ") end
          #end
          μνλσ = 1 + (σσ-pσ) + nσ*(λλ-pλ) + nσ*nλ*(νν-pν) + nσ*nλ*nν*(μμ-pμ)
          jlswap = (μμ < λλ) || (μμ == λλ && νν < σσ)
          
          eri = eri_batch[μνλσ] 
             
          if Base.abs_float(eri) < cutoff
            #if do_continue_print println("CONTINUE SCREEN") end
            continue 
          elseif jlswap && ish == ksh && jsh == lsh 
            continue
          end

          μ = max(μμ, λλ)
          ν = !jlswap ? νν : σσ
          λ = min(μμ, λλ)
          σ = !jlswap ? σσ : νν
      
          #println("QUARTET($ish, $jsh, $ksh, $lsh): $eri")
          #println("ERI($μ, $ν, $λ, $σ) = $eri") 
      
          eri *= (μ == ν) ? 0.5 : 1.0 
          eri *= (λ == σ) ? 0.5 : 1.0
          eri *= ((μ == λ) && (ν == σ)) ? 0.5 : 1.0

          F_priv[λ,σ] += 4.0 * D[μ,ν] * eri
          F_priv[μ,ν] += 4.0 * D[λ,σ] * eri
          F_priv[μ,λ] -= D[ν,σ] * eri
          F_priv[μ,σ] -= D[ν,λ] * eri
          F_priv[max(ν,λ), min(ν,λ)] -= D[μ,σ] * eri
          F_priv[max(ν,σ), min(ν,σ)] -= D[μ,λ] * eri
        end
      end
    end
  end
end

#=
"""
	 iteration(F::Matrix{Float64}, D::Matrix{Float64}, H::Matrix{Float64}, ortho::Matrix{Float64})
Summary
======
Perform single SCF iteration.

Arguments
======
D = Current iteration's Density Matrix

H = One-electron Hamiltonian Matrix

ortho = Symmetric Orthogonalization Matrix
"""
=#
function iteration(F_μν::Matrix{Float64}, D::Matrix{Float64},
  C::Matrix{Float64}, H::Matrix{Float64}, F_eval::Vector{Float64},
  F_evec::Matrix{Float64}, workspace_a::Matrix{Float64}, 
  workspace_b::Matrix{Float64}, ortho::Matrix{Float64}, 
  basis_sets::CalculationBasisSets, iter::Int, debug::Bool)
  basis = basis_sets.primary
  comm=MPI.COMM_WORLD
 
  transpose!(workspace_b, LinearAlgebra.Hermitian(ortho)) 

  #== obtain new orbital coefficients ==#
  BLAS.symm!('L', 'U', 1.0, workspace_b, F_μν, 0.0, workspace_a)
  BLAS.gemm!('N', 'N', 1.0, workspace_a, ortho, 0.0, workspace_b)

  F_eval[:], F_evec[:,:] = eigen!(LinearAlgebra.Hermitian(workspace_b)) 
  
  #@views F_evec .= F_evec[:,sortperm(F_eval)] #sort evecs according to sorted evals

  if debug && MPI.Comm_rank(comm) == 0
    h5write("debug.h5","RHF/Iteration-$iter/F_evec", F_evec)
  end

  #C .= ortho*F_evec
  BLAS.symm!('L', 'U', 1.0, ortho, F_evec, 0.0, C)
  
  if debug && MPI.Comm_rank(comm) == 0
    h5write("debug.h5","RHF/Iteration-$iter/C", C)
  end

  #== build new density matrix ==#
  nocc = basis.nels >> 1
  norb = basis.norb

  #fill!(D, 0.0)
  for i in 1:basis.norb, j in 1:basis.norb
    D[i,j] = 2.0*BLAS.dot(nocc,pointer(C,i),norb,pointer(C,j),norb)
  end
 
  #== compute new SCF energy ==#
  #EHF1 = LinearAlgebra.dot(D, F_μν)
  #EHF2 = LinearAlgebra.dot(D, H)
  EHF1 = LinearAlgebra.BLAS.dot(length(D), D, 1, F_μν, 1)
  EHF2 = LinearAlgebra.BLAS.dot(length(D), D, 1, H, 1)
  E_elec = (EHF1 + EHF2)/2.0
  
  if debug && MPI.Comm_rank(comm) == 0
    h5write("debug.h5","RHF/Iteration-$iter/D", D)
    h5write("debug.h5","RHF/Iteration-$iter/E/EHF1", EHF1)
    h5write("debug.h5","RHF/Iteration-$iter/E/EHF2", EHF2)
    h5write("debug.h5","RHF/Iteration-$iter/E/EHF", E_elec)
  end

  return E_elec, F_eval
end

function check_auxillary_basis_is_provided(do_density_fitting, basis_sets)
  if do_density_fitting && isnothing(basis_sets.auxillary)
    error("Density Fitting scf_type selected but no auxillary basis set was found. Check your input: model.auxiliary_basis value")
  end
end
