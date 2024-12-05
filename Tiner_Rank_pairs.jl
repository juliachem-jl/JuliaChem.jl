function get_rank_round_robin(n_ranks)

   
    half_of_ranks = n_ranks÷2
    ranks = collect(0:n_ranks-1)
    n_rounds = n_ranks-1

    round_rank_partners = zeros(Int, n_rounds, n_ranks)

    for round in 1:n_rounds
        for i in 0:half_of_ranks-1
            left_rank = ranks[i+1]
            right_rank = ranks[(half_of_ranks*2)-i]
            # println("i = $i ($left_rank,$right_rank)")
            round_rank_partners[round, left_rank+1] = right_rank
            round_rank_partners[round, right_rank+1] = left_rank
        end
        if n_ranks %2 != 0
            end_rank = ranks[end] # paired with itself 
            round_rank_partners[round, end_rank+1] = end_rank
        end
    
        #shift elements to the right 
        temp = ranks[end]
        ranks[end] = ranks[end-1]
        for j in n_ranks-1:-1:2
            ranks[j+1] = ranks[j]
        end
        ranks[2] = temp

    end
    return round_rank_partners

end

display(get_rank_round_robin(5))