using MPI


function reduce_B_this_rank(B, length_of_B, rank, other_rank)
    comm = MPI.COMM_WORLD
    max_value = (2^31 - 1)-1
    top_index = max_value
    start_index = 1
    while true
        top_index = min(top_index, length_of_B)
        MPI.Recv!(view(B, start_index:top_index), comm; source=other_rank)
        if top_index == length_of_B
            break
        end
        start_index = top_index + 1
        top_index += max_value

    end    
end

function reduce_B_other_rank(B, length_of_B, rank, other_rank)
    comm = MPI.COMM_WORLD
    max_value = (2^31 - 1) -1
    top_index = max_value
    start_index = 1
    while true
        top_index = min(top_index, length_of_B)
        MPI.Send(view(B, start_index:top_index), comm; dest=other_rank)
        if top_index == length_of_B
            break
        end
        start_index = top_index + 1
        top_index += max_value
    end    
end

# function main()
#     comm = MPI.COMM_WORLD
#     this_rank = MPI.Comm_rank(comm)
#     n_ranks = MPI.Comm_size(comm)

#     Q = 80# 10350 
#     pq = 100
#     Q_per_rank = Q ÷ n_ranks

#     rank_Q_ranges = []
#     for i in 0:n_ranks-2
#         push!(rank_Q_ranges, (i)*Q_per_rank+1: ((i+1)*Q_per_rank)-1)
#     end
#     push!(rank_Q_ranges, (n_ranks-1)*Q_per_rank:Q)
    
#     rank_Q_length = length(rank_Q_ranges[this_rank+1])

#     println("Rank: ", this_rank, " Q_range: ", rank_Q_ranges[this_rank+1], " range length", rank_Q_length, "\n")

#     B = zeros(rank_Q_length, pq)
#     B .= this_rank+1

#     max_Q_length = maximum([length(x) for x in rank_Q_ranges]) 

#     temp_B_buffer = zeros(max_Q_length, pq)

#     for other_rank in 0:n_ranks-1
#         if other_rank == this_rank
#             for send_rank in 0:n_ranks-1
#                 if send_rank != this_rank
#                     reduce_B_this_rank(B, rank_Q_length*pq, this_rank, send_rank)
#                 end
#             end
#         else
#             temp_B_buffer .= other_rank+1
#             reduce_B_other_rank(temp_B_buffer, rank_Q_length*pq, this_rank, other_rank)
#         end
#     end

#     println("Rank: ", this_rank, " B: ", B[1:5 , 1:5])
# end

function main()
    comm = MPI.COMM_WORLD
    this_rank = MPI.Comm_rank(comm)
    n_ranks = MPI.Comm_size(comm)

    Q = 10350 ÷ n_ranks
    p = 274086

    B = ones( Q,  p)
    BB = zeros( Q,  p)
    send_to_rank = 0 
    if this_rank == 0
        send_to_rank = 1
    elseif this_rank == 1
        send_to_rank = 0
    elseif this_rank == 2
        send_to_rank = 3
    elseif this_rank == 3
        send_to_rank = 2
    end

    send_1_time = @elapsed begin 
        if this_rank % 2 == 0
            reduce_B_other_rank(B, Q*p, this_rank, send_to_rank)
        else
            reduce_B_this_rank(B, Q*p, this_rank, send_to_rank)
            # BB += B
        end
    end
    #print the size of B in GB 
    println("Rank: ", this_rank, " B size: ", sizeof(B)/1024^3, " GB")
    println("Rank: ", this_rank, " Send 1 time: ", send_1_time)
    
    send_2_time = @elapsed begin 
        if this_rank % 2 == 0
            reduce_B_this_rank(B, Q*p, this_rank, send_to_rank)
            # BB += B
        else
            reduce_B_other_rank(B, Q*p, this_rank, send_to_rank)
        end
    end


    println("Rank: ", this_rank, " BB: ", BB[1:5 , 1:5])



end

MPI.Init()

main()

MPI.Finalize()

