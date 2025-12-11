include("DynamicLoad.jl")


function main()
    println("starting main\n")
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)
    A = 1738
    batch_size = A
    
    two_center_integrals = zeros(Float64, A, A)
    cartesian_indices = CartesianIndices(two_center_integrals)
    n_indicies = length(cartesian_indices)
    mutex_mpi_worker = Base.Threads.ReentrantLock()
    index_lock = Base.Threads.ReentrantLock()
    n_threads = Threads.nthreads()

    task_top_index = [n_indicies]

    for r in 0:n_ranks-1
        for t in 1:n_threads
            if r == 0 && t == 1 && n_ranks > 1
                continue
            end
            task_top_index[1] -= batch_size + 1
        end
    end

    if rank > 0
        task_top_index[1] = 1000000000 # not a real index this value should never be used
    end


    ## all threads get pre determined first index to proces
    
    println("task_top_index: $(task_top_index[1])\n")
    
    @sync for thread in 1:Threads.nthreads()
        Threads.@spawn begin
            threadid = Threads.threadid()
            if rank == 0 && threadid == 1 && n_ranks > 1
                setup_integral_coordinator(task_top_index, batch_size, n_ranks, n_threads, mutex_mpi_worker, index_lock)
            else 
                worker_thread_number = 0                
                if n_ranks > 1
                    worker_thread_number = threadid + (rank*n_threads-2)                
                else
                    worker_thread_number = threadid + (rank*n_threads-1)    
                end
                first_task = n_indicies - (batch_size*worker_thread_number) - worker_thread_number
                ij_index = first_task
                while ij_index > 0
                    for ij in ij_index:-1:max(1, ij_index - batch_size)
                        shell_index = cartesian_indices[ij]
                        two_center_integrals[shell_index] += 1.0
                    end
                    ij_index = get_next_task(mutex_mpi_worker, task_top_index, batch_size, index_lock, ij_index)
                end
            end        
        end
    end

    #clean up any outstanding requests for work
    # ismessage = true
    # recv_mesg = [0]
    #     if rank == 0
    #     recv_mesg = [0,0,0]    
    # end
    # while ismessage
    #     ismessage, status = MPI.Iprobe(-2, -1, comm)
    #     if ismessage
    #         rreq = MPI.Recv!(recv_mesg, status.source, status.tag, comm) 
    #         println("there was an outstanding request to $rank from rank: $(status.source) thread: $(status.tag) msg $(recv_mesg)")
    #     end
    # end

    MPI.Barrier(comm)
   

    MPI.Allreduce!(two_center_integrals, MPI.SUM, comm)
    if rank == 0 
        failed = false
        for index in cartesian_indices
            if two_center_integrals[index] != 1.0
                println("TCI[$(index[1]), $(index[2])] = $(two_center_integrals[index])\n")
                failed = true
            end
        end
        if failed
            println("FAILED\n")
        else
            println("PASSED\n")
        end    
    end
end
MPI.Init()
try
    main()
catch ex
    println(ex)
end
MPI.Finalize()