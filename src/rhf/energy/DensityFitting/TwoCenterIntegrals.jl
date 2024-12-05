using Base.Threads
using LinearAlgebra
using JuliaChem.Shared.Constants.SCF_Keywords
using JuliaChem.Shared
two_center_integral_tag = 2000

function calculate_two_center_intgrals(jeri_engine_thread::Vector{T}, basis_sets, scf_options::SCFOptions) where {T<:DFRHFTEIEngine}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    aux_basis_function_count = basis_sets.auxillary.norb
    two_center_integrals = zeros(Float64, aux_basis_function_count, aux_basis_function_count)
    auxilliary_basis_shell_count = length(basis_sets.auxillary)
    cartesian_indices = CartesianIndices((auxilliary_basis_shell_count, auxilliary_basis_shell_count))
    
    max_nbas = max_number_of_basis_functions(basis_sets.auxillary)
    n_threads = Threads.nthreads()
    thead_integral_buffer = [zeros(Float64, max_nbas^2) for i in 1:n_threads]

    if scf_options.load == "sequential"
        calculate_two_center_integrals_sequential!(two_center_integrals, cartesian_indices, jeri_engine_thread[1], thead_integral_buffer[1], basis_sets)
    else
        if rank == 0 && scf_options.load != "static"
            println("integral threading load type: $(scf_options.load) not supported, defaulting to static")
        end
        calculate_two_center_integrals_static!(two_center_integrals, cartesian_indices, jeri_engine_thread, thead_integral_buffer, basis_sets)
    end
    # print_two_center_integrals(two_center_integrals)
    return two_center_integrals
end

@inline function calculate_two_center_integrals_kernel!(two_center_integrals,
    engine,
    cartesian_index,
    basis_sets,
    integral_buffer)
    shell_1_index = cartesian_index[1]
    shell_2_index = cartesian_index[2]

    if shell_2_index > shell_1_index #the top triangle of the symmetric matrix does not need to be calculated
        return
    end

    shell_1 = basis_sets.auxillary.shells[shell_1_index]
    shell_1_basis_count = shell_1.nbas
    bf_1_pos = shell_1.pos

    shell_2 = basis_sets.auxillary.shells[shell_2_index]
    shell_2_basis_count = shell_2.nbas
    bf_2_pos = shell_2.pos

    JERI.compute_two_center_eri_block(engine, integral_buffer, shell_1_index - 1, shell_2_index - 1, shell_1_basis_count, shell_2_basis_count)
    copy_integral_results!(two_center_integrals, integral_buffer, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
    axial_normalization_factor(two_center_integrals, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count, bf_1_pos, bf_2_pos)
end

@inline function calculate_two_center_integrals_sequential!(two_center_integrals, cartesian_indices, engine, integral_buffer, basis_sets)
    for cartesian_index in cartesian_indices
        calculate_two_center_integrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
    end
end

@inline function calculate_two_center_integrals_static!(two_center_integrals, cartesian_indices, jeri_engine_thread, thead_integral_buffer, basis_sets)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)
    nthreads = Threads.nthreads()
    aux_basis_length = length(basis_sets.auxillary)
    

    load_balance_indicies = [static_load_rank_indicies(rank_index, n_ranks, basis_sets) for rank_index in 0:n_ranks-1]
    rank_shell_indicies = load_balance_indicies[rank+1][1] 
    rank_basis_indicies = load_balance_indicies[rank+1][2] 

    rank_number_of_shells = length(rank_shell_indicies)
    n_indicies_per_thread = rank_number_of_shells÷nthreads

    Threads.@sync for thread in 1:nthreads
        Threads.@spawn begin     
            thread_index_offset = static_load_thread_index_offset(thread, n_indicies_per_thread)
            n_shells_to_process = static_load_thread_shell_to_process_count(thread, nthreads, rank_number_of_shells, n_indicies_per_thread)
            for shell_index in 1:n_shells_to_process
                B = rank_shell_indicies[shell_index + thread_index_offset]
                for A in 1:aux_basis_length
                    cartesian_index = CartesianIndex(A, B)
                    engine = jeri_engine_thread[thread]
                    integral_buffer = thead_integral_buffer[thread]
                    calculate_two_center_integrals_kernel!(two_center_integrals, engine, cartesian_index, basis_sets, integral_buffer)
                end
            end    
        end
    end
    if n_ranks > 1
        # MPI.Allreduce!(two_center_integrals, MPI.SUM, comm)
        gather_and_reduce_two_center_integrals(two_center_integrals, load_balance_indicies, rank_basis_indicies, comm)
    end
end

function gather_and_reduce_two_center_integrals(two_center_integrals, load_balance_indicies, rank_basis_indicies, comm)
    rank = MPI.Comm_rank(comm)

    number_of_aux_basis_funtions = size(two_center_integrals, 1)
    aux_basis_indicies_per_rank = [length(x[2]) for x in load_balance_indicies] # number of basis functions calculated on each 
    rank_indicies = [x[2] for x in load_balance_indicies] #basis function indicies calculated on each 
    indicies_per_rank = aux_basis_indicies_per_rank.*number_of_aux_basis_funtions
    # two_center_integral_buff = MPI.VBuffer(two_center_integrals, indicies_per_rank) # Buffer set up with the correct size for each rank
    

    MPI.Barrier(comm)
    gather_two_center_integrals(two_center_integrals, aux_basis_indicies_per_rank, rank_indicies, comm)
    # MPI.Allgatherv!(two_center_integrals[ :,rank_basis_indicies], two_center_integral_buff, comm) # gather the data from each rank into the Buffer
    # reorder_mpi_gathered_matrix(two_center_integrals, rank_indicies, set_data_2D!, set_temp_2D!, zeros(Float64, number_of_aux_basis_funtions))
end


function gather_two_center_integrals(two_center_integrals, indicies_per_rank, rank_index_ranges, comm)
    n_ranks = MPI.Comm_size(comm)
    this_rank = MPI.Comm_rank(comm)
    Q = size(two_center_integrals, 1)
    max_length = (2^31 - 1) -1
 
    if this_rank == 0
        for send_rank in 1:n_ranks-1
            send_rank_n_integrals = indicies_per_rank[send_rank+1]*Q
            if send_rank_n_integrals < max_length
                send_rank_data = view(two_center_integrals, :, rank_index_ranges[send_rank+1])
                buff = MPI.Buffer(send_rank_data, send_rank_n_integrals, MPI.Datatype(Float64))
                MPI.Recv!(buff, comm; source=send_rank, tag=send_rank)
            else
                n_inidices_per_chunk = send_rank_n_integrals ÷ max_length
                for i in 1:n_inidices_per_chunk+1
                    start_index = (i-1)*max_length + 1
                    end_index = min(i*max_length, send_rank_n_integrals)
                    if start_index > send_rank_n_integrals
                        break
                    end
                    rank_data = view(view(two_center_integrals, :, rank_index_ranges[send_rank+1]), start_index:end_index)
                    n_values = length(rank_data)
                    buff = MPI.Buffer(rank_data, n_values, MPI.Datatype(Float64))
                    MPI.Recv!(buff, comm; source=send_rank, tag=send_rank)
                end
            end
        end
    else 
        rank_n_integrals = indicies_per_rank[this_rank+1]*Q
        if rank_n_integrals < max_length
            rank_data = view(two_center_integrals, :, rank_index_ranges[this_rank+1])
            buff = MPI.Buffer(rank_data, rank_n_integrals, MPI.Datatype(Float64))
            MPI.Send(buff, comm; dest=0, tag=this_rank)
        else
            n_inidices_per_chunk = rank_n_integrals ÷ max_length
            for i in 1:n_inidices_per_chunk+1

                start_index = (i-1)*max_length + 1
                end_index = min(i*max_length, rank_n_integrals)
                rank_data = view(view(two_center_integrals, :, rank_index_ranges[this_rank+1]), start_index:end_index)

                if start_index > rank_n_integrals
                    break
                end
                n_values = length(rank_data)
                buff = MPI.Buffer(rank_data, n_values, MPI.Datatype(Float64))
                MPI.Send(buff,comm; dest=0, tag=this_rank)
            end
        end        
    end
end

@inline function run_two_center_integrals_worker(two_center_integrals,
    cartesian_indices,
    batch_size,
    jeri_engine_thread,
    thead_integral_buffer, basis_sets, top_index, mutex_mpi_worker, n_indicies, thread_indicies_processed, thread)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_threads = Threads.nthreads()
    n_ranks = MPI.Comm_size(comm)

    
    lock(mutex_mpi_worker)
        worker_thread_number = get_worker_thread_number(thread, rank, n_threads, n_ranks)
        ij_index = get_first_task(n_indicies, batch_size, worker_thread_number)
    unlock(mutex_mpi_worker)
    
    while ij_index > 0
        do_two_center_integral_batch(two_center_integrals,
        ij_index,
        batch_size,
        cartesian_indices,
        jeri_engine_thread[thread],
        thead_integral_buffer[thread], basis_sets)
        ij_index = get_next_task(mutex_mpi_worker, top_index, batch_size, thread, two_center_integral_tag)
    end
end

@inline function do_two_center_integral_batch(two_center_integrals,
    top_index,
    batch_size,
    cartesian_indices,
    engine,
    integral_buffer, basis_sets)
    for ij in top_index:-1:(max(1, top_index - batch_size))
        shell_index = cartesian_indices[ij]
        calculate_two_center_integrals_kernel!(two_center_integrals, engine, shell_index, basis_sets, integral_buffer)
    end
end



@inline function copy_integral_results!(two_center_integrals, values, shell_1, shell_2, shell_1_basis_count, shell_2_basis_count)
    temp_index = 1
    for i in shell_1.pos:shell_1.pos+shell_1_basis_count-1
        for j in shell_2.pos:shell_2.pos+shell_2_basis_count-1
            if i >= j # makes sure we don't put any values onto the top triangle which happens for some reason sometimes 
                two_center_integrals[i, j] = values[temp_index]
            else
                two_center_integrals[i, j] = 0.0
            end
            temp_index += 1
        end
    end
end

function print_two_center_integrals(two_center_integrals)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    println("two center integrals\n")
    i = 0
    io = open(joinpath(@__DIR__,  "./two_center_integrals_out-rank-$rank.txt"), "w+")
    for index in CartesianIndices(two_center_integrals)
        write(io,"2-ERI[$(index[1]),$(index[2])] = $(two_center_integrals[index])\n")
        i += 1
        # if i > 10000 
        #     break
        # end
    end
    close(io)
end



function broadcast_two_center_integrals(two_center_integrals)
    #max value of int32 
    comm = MPI.COMM_WORLD
    max_length = (2^31 - 1) -1
    if length(two_center_integrals) < max_length
        MPI.Bcast!(two_center_integrals, 0, comm)
        return
    end
    #loop over the array and send in chunks of max value of int32
    number_of_chunks = length(two_center_integrals) ÷ max_length
    for i in 1:number_of_chunks+1
        start_index = (i-1)*max_length + 1
        end_index = min(i*max_length, length(two_center_integrals))
        if start_index > length(two_center_integrals)
            break
        end
        MPI.Bcast!(two_center_integrals[start_index:end_index], 0, comm)
    end    
end