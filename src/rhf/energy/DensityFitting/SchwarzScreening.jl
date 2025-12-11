""" Schwarz screening for DF-RHF as described in Huang et al. 2020 https://doi.org/10.1063/1.5129452
if σ is a threshold on the size of (pq|P), 
then (pq|P) can be neglected if (pq∣pq) < σ^2 / maxP(P∣P) 
where max_P(P|P) is the maximum value of (P|P) over all P
Arguments
=========
σ = 10^(−12) is used in huang et. al 2020 screening parameter
"""
function schwarz_screen_itegrals_df(scf_data, σ, max_P_P, basis_sets, jeri_engine_thread)

    basis = basis_sets.primary
    basis_set_length = length(basis)
    shell_screen_matrix = zeros(Bool,basis_set_length, basis_set_length) # true means keep the shell pair, false means it is screened
    basis_function_screen_matrix = zeros(Bool,scf_data.μ, scf_data.μ) # true means keep the basis function pair, false means it is screened

    n_shell_indicies = basis_set_length * (basis_set_length + 1) ÷ 2 # # of triangular shell pairs (pq)
    max_am = max_ang_mom(basis) 
    batch_size = eri_quartet_batch_size(max_am)
    nthreads = Threads.nthreads()
    eri_quartet_batch_thread = [ Vector{Float64}(undef, batch_size) for thread in 1:nthreads ]

    threshold = (σ^2) / max_P_P #10.0^-10 hardcode for comparison with EXESS GPU DF-RHF
    Threads.@sync for thread in 1:nthreads
        Threads.@spawn begin
            for index in thread:nthreads:n_shell_indicies
                bra_pair = index
                ket_pair = index
                ish = decompose(index)
                jsh = index - triangular_index(ish)

                μ_shell = basis[ish]
                ν_shell = basis[jsh]
            
                nμ = μ_shell.nbas
                nν = ν_shell.nbas

                μ_position = μ_shell.pos
                ν_position = ν_shell.pos

                eri_quartet_batch_thread[thread] .= 0.0
                JERI.compute_eri_block(jeri_engine_thread[thread], eri_quartet_batch_thread[thread],
                    ish, jsh, ish, jsh, bra_pair, ket_pair, nμ * nν, nμ * nν)

                axial_normalization_factor(eri_quartet_batch_thread[thread], μ_shell, ν_shell, μ_shell, ν_shell, nμ, nν, nμ, nν)

                shell_pair_contracted = sum(eri_quartet_batch_thread[thread])


                shell_screen_matrix[ish, jsh] = !(Base.abs_float(shell_pair_contracted) < threshold)
                shell_screen_matrix[jsh, ish] = shell_screen_matrix[ish, jsh]

                if shell_screen_matrix[ish, jsh] == false # if the shell pair is screened, then screen all the basis function pairs
                    for μμ::Int64 in μ_position:(μ_position+nμ-1)
                        for νν::Int64 in ν_position:(ν_position+nν-1)
                            basis_function_screen_matrix[μμ, νν] = false
                            basis_function_screen_matrix[νν, μμ] = false
                        end
                    end
                else #screen individual basis functions pairs within the shell pair
                    for μμ::Int64 in μ_position:(μ_position+nμ-1)
                        for νν::Int64 in ν_position:(ν_position+nν-1)
                            μνμν = 1 + (νν - ν_position) + nν * (μμ - μ_position) + nν * nμ * (νν - ν_position) + nν * nμ * nν * (μμ - μ_position)
                            eri = eri_quartet_batch_thread[thread][μνμν]
                            basis_function_screen_matrix[μμ, νν] = !(Base.abs_float(eri) < threshold)
                            basis_function_screen_matrix[νν, μμ] = basis_function_screen_matrix[μμ, νν]
                        end
                    end
                end
            end
        end # end of thread spawn
    end # end of thread sync 
    sparse_pq_index_map = zeros(Int64, scf_data.μ, scf_data.μ)
    sparse_index = 1
    for pp::Int64 in 1:scf_data.μ
        for qq::Int64 in 1:scf_data.μ
            if basis_function_screen_matrix[qq,pp] == true
                sparse_pq_index_map[qq,pp] = sparse_index
                sparse_index += 1                
            end
        end
    end
    return shell_screen_matrix, basis_function_screen_matrix, sparse_pq_index_map
end


function get_max_P_P(two_center_integrals)
    P = size(two_center_integrals)[1]
    maxP = floatmin(Float64) # smallest possible float
    for p in 1:P
        if Base.abs_float(two_center_integrals[p,p]) > maxP
            maxP = abs(two_center_integrals[p,p])
        end
    end
    return maxP
end

function setup_unscreened_screening_matricies(basis_sets, scf_data)
    basis = basis_sets.primary
    basis_set_length = length(basis)
    scf_data.screening_data.shell_screen_matrix = ones(Bool,basis_set_length, basis_set_length) # true means keep the shell pair, false means it is screened
    scf_data.screening_data.basis_function_screen_matrix = ones(Bool,scf_data.μ, scf_data.μ) # true means keep the basis function pair, false means it is screened
    scf_data.screening_data.screened_indices_count = scf_data.μ^2
    scf_data.screening_data.sparse_pq_index_map = zeros(Int64, (scf_data.μ, scf_data.μ))

    Threads.@threads for pp in 1:scf_data.μ
      for qq in 1:scf_data.μ
        #2D index to 1D index row major (why is this row major?)
        scf_data.screening_data.sparse_pq_index_map[pp, qq] = pp + (qq - 1) * scf_data.μ
      end
    end
end

export schwarz_screen_itegrals_df, get_max_P_P, setup_unscreened_screening_matricies