function main()
    n_indicies = 2025
    batch_size = 45

    n_ranks = 2
    n_threads = 2

    top_indicies = zeros(Int64, n_ranks, n_threads)

    for rank in 0:n_ranks-1
        for thread in 1:n_threads
            thread_number = (thread + (rank*n_threads-1))
            top_indicies[rank+1, thread] = n_indicies - (batch_size*thread_number) - thread_number
        end
    end
    display(top_indicies)
    println()
    display(top_indicies.-batch_size)
end

main()