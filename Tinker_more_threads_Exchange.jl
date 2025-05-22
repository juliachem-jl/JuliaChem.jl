using MKL
using LinearAlgebra 
using Base.Threads

const BlasInt = LinearAlgebra.BlasInt
const libblastrampoline = LinearAlgebra.libblastrampoline


function get_triangle_matrix_length(n)::Int
    return n * (n + 1) ÷ 2
end

function do_symmetric_K(W, exchange, p, n_occ, Q, blocks_wide)

    n_threads = Threads.nthreads()
    lower_triangle_length_without_diagonal = get_triangle_matrix_length(1)
    # blocks_wide = 11
    lower_triangle_length = get_triangle_matrix_length(blocks_wide)
    # blocks_wide = 2
    # while true 
    #     new_lower_triangle_length = get_triangle_matrix_length(blocks_wide+1)
    #     if new_lower_triangle_length > n_threads
    #         break
    #     end
    #     lower_triangle_length_without_diagonal = lower_triangle_length
    #     lower_triangle_length = new_lower_triangle_length
    #     blocks_wide += 1
        
    # end

    remaining_threads =  lower_triangle_length - n_threads

    # break up the K work into blocks and allocate temporary memory for it 
    number_of_blocks = 1
    K_blocks = Vector{Array{Float64, 2}}(undef, 0)

    K_block_width = p ÷ blocks_wide
    p_ranges = Array{UnitRange{Int}}(undef, 0)
    q_ranges = Array{UnitRange{Int}}(undef, 0)

    for iii in 1:blocks_wide
        for jjj in 1:iii-1
            p_range = (iii-1)*K_block_width+1:iii*K_block_width
            q_range = (jjj-1)*K_block_width+1:jjj*K_block_width
            push!(p_ranges, p_range)
            push!(q_ranges, q_range)
            push!(K_blocks, zeros(Float64, K_block_width, K_block_width))
            number_of_blocks+=1
        end
    end

    
    diagonal_start_index = number_of_blocks
    for iii in 1:blocks_wide
        jjj = iii

        p_range = (iii-1)*K_block_width+1:iii*K_block_width
        q_range = (jjj-1)*K_block_width+1:jjj*K_block_width

        #divide range pair into four blocks 
        
        top_left_p_range = p_range[1:K_block_width÷2]
        top_left_q_range = q_range[1:K_block_width÷2]
        top_right_p_range = p_range[K_block_width÷2+1:end]
        top_right_q_range = q_range[1:K_block_width÷2]

        bottom_left_p_range = p_range[1:K_block_width÷2]
        bottom_left_q_range = q_range[K_block_width÷2+1:end]

        bottom_right_p_range = p_range[K_block_width÷2+1:end]
        bottom_right_q_range = q_range[K_block_width÷2+1:end]

        push!(p_ranges, top_left_p_range)
        push!(q_ranges, top_left_q_range)
        push!(K_blocks, zeros(Float64, length(top_left_p_range), length(top_left_q_range)))
        number_of_blocks+=1
        # p_ranges[number_of_blocks] = top_right_p_range
        # q_ranges[number_of_blocks] = top_right_q_range
        # K_blocks[number_of_blocks] = zeros(Float64, length(top_right_p_range), length(top_right_q_range)) 
        # number_of_blocks+=1
        push!(p_ranges, bottom_left_p_range)
        push!(q_ranges, bottom_left_q_range)
        push!(K_blocks, zeros(Float64, length(bottom_left_p_range), length(bottom_left_q_range)))
        number_of_blocks+=1

        push!(p_ranges, bottom_right_p_range)
        push!(q_ranges, bottom_right_q_range)
        push!(K_blocks, zeros(Float64, length(bottom_right_p_range), length(bottom_right_q_range)))
        number_of_blocks+=1

    end
    number_of_blocks = length(K_blocks)
    println("lower_triangle_length: ", lower_triangle_length)
    println("diagonal_start_index: ", diagonal_start_index)
    println("number_of_blocks: ", number_of_blocks)


   
    alpha = 1.0 
    beta = 0.0
    linear_indices = LinearIndices(W)
    transA = true
    transB = false
    n_threads = Threads.nthreads()
    thread_times = zeros(Float64, n_threads)
    K = n_occ * Q
    block_times = zeros(Float64, number_of_blocks)
    block_start_times = zeros(Float64, number_of_blocks)
    start_time = time()
    @time begin 
        W_reshape = reshape(W, Q*n_occ, p)
        # start_time = time()
        next_block = n_threads + 1
        index_lock =  Base.Threads.ReentrantLock()
        Threads.@sync for thread in 1:n_threads
            Threads.@spawn begin
                # index = thread
                # Threads.@threads for block_index in 1:number_of_blocks
                    # for block_index in thread:n_threads:number_of_blocks            

                block_index = thread
                while block_index <= number_of_blocks               
                    block_start_times[block_index] = time()
                    # p_range = p_ranges[block_index]
                    # q_range = q_ranges[block_index] 
                
                    # A_view = view(W_reshape, :, p_range)
                    # B_view = view(W_reshape, :, q_range)
                    # p_range = (pp-1)*K_block_width+1:pp*K_block_width
                    p_start = p_ranges[block_index][1]
                    q_start = q_ranges[block_index][1]

                    M = K_block_width
                    N = K_block_width

                    if block_index >= diagonal_start_index
                        M = size(K_blocks[block_index], 1)
                        N = size(K_blocks[block_index], 2)
                    end

                    A_ptr = pointer(W, linear_indices[1, 1, p_start])
                    B_ptr = pointer(W, linear_indices[1, 1, q_start])
                    C_ptr = pointer(K_blocks[block_index], 1)
                    call_gemm!(Val(transA), Val(transB), M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr)
                    block_end_time = time()
                    block_times[block_index] = block_end_time - block_start_times[block_index]
                    # LinearAlgebra.BLAS.gemm!('T', 'N', 1.0, A_view, B_view, 0.0, K_blocks[block_index])

                    lock(index_lock) do
                        block_index = next_block
                        next_block += 1
                    end
                end
            end 
        end
                # display(thread_times)
        Threads.@threads for block_index in 1:number_of_blocks
            p_range = p_ranges[block_index]
            q_range = q_ranges[block_index]

            # K_blocks[block_index] = transpose(K_blocks[block_index])
            view(exchange, p_range, q_range) .= K_blocks[block_index]
            if p_range[1] != q_range[1]
                view(exchange, q_range, p_range) .= transpose(K_blocks[block_index])
            end
        end
            # do the remaining region of exchange 
            # BLAS.set_num_threads(Threads.nthreads())
        if p % blocks_wide != 0
            remainder_block = Array{Float64, 2}(undef, p % blocks_wide, p)


            A_view = view(W_reshape, :, p-(p%blocks_wide)+1:p)
            BLAS.gemm!('T', 'N', 1.0, A_view, W_reshape, 0.0, remainder_block)
            #transpose the remainder 
            view(exchange, p-(p%blocks_wide)+1:p, :) .= remainder_block
            view(exchange, :, p-(p%blocks_wide)+1:p) .= transpose(remainder_block)
        end
    end
    
    # display(block_start_times)
    # display(block_times)

end


#todo move this to a BLAS shared file
function call_gemm!(transA::Val, transB::Val,
    M::Int, N::Int, K::Int,
    alpha::Float64, A::Ptr{Float64}, B::Ptr{Float64},
    beta::Float64, C::Ptr{Float64})

    # Convert our compile-time transpose marker to a char for BLAS
    convtrans(V::Val{false}) = 'N'
    convtrans(V::Val{true}) = 'T'

    if transA == Val(false)
        lda = M
    else
        lda = K
    end
    if transB == Val(false)
        ldb = K
    else
        ldb = N
    end
    ldc = M

    ccall((:dgemm_64_, BLAS.libblas), Nothing,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ref{BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
            Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
            Ref{BlasInt}),
        convtrans(transA), convtrans(transB), M, N, K,
        alpha, A, lda, B, ldb, beta, C, ldc)
end



function main()

    p = 1400
    n_occ = p ÷ 4  
    Q = p*4

    W = rand(Float64, Q, n_occ, p)
    K = zeros(Float64, p, p) 
    K_no_sym = zeros(Float64, p, p)
    

    println("doing symmetric K")
    LinearAlgebra.BLAS.set_num_threads(1)

    for blocks_wide in 8:12
        println("blocks wide = ", blocks_wide)
        do_symmetric_K(W, K, p, n_occ, Q, blocks_wide)
        do_symmetric_K(W, K, p, n_occ, Q, blocks_wide)
        do_symmetric_K(W, K, p, n_occ, Q, blocks_wide)
    end


    W_reshape = reshape(W, Q*n_occ, p)
    BLAS.set_num_threads(Threads.nthreads())
    @time BLAS.gemm!('T', 'N', 1.0, W_reshape, W_reshape, 0.0, K_no_sym)
    @time BLAS.gemm!('T', 'N', 1.0, W_reshape, W_reshape, 0.0, K_no_sym)
    @time BLAS.gemm!('T', 'N', 1.0, W_reshape, W_reshape, 0.0, K_no_sym)

    diff = K - K_no_sym
    println("Max difference: ", maximum(abs.(diff)))


end

main()
