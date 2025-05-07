using AMDGPU
using HDF5
using LinearAlgebra



function cpu_gemm(W)
    # Perform the matrix multiplication on the CPU
    Q = size(W, 1)
    n_occ = size(W, 2)
    p = size(W, 3)    
    K = zeros(Float64, p,p)
    LinearAlgebra.BLAS.gemm!('T','N',-1.0, 
    reshape(W, Q *n_occ,p) , reshape(W, Q *n_occ,p), 0.0, K)
    return K
end

function gpu_gemm_no_sym(W)

    # hipstream =  AMDGPU.HIPStream(:high)
    # AMDGPU.stream!(hipstream)

    # Perform the matrix multiplication on the GPU
    Q = size(W, 1)
    n_occ = size(W, 2)
    p = size(W, 3)    
    K = AMDGPU.zeros(Float64, p,p)
    # Perform the matrix multiplication on the GPU
    gemm_time = @elapsed begin 
        AMDGPU.rocBLAS.gemm!('T','N',-1.0, 
        reshape(W, Q *n_occ,p) , reshape(W, Q *n_occ,p), 0.0, K )
        AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    end
    # println("gemm_time: ", gemm_time)
    return K, gemm_time
end

function get_triangle_matrix_length(n)::Int
    return n * (n + 1) ÷ 2
end

function gpu_gemm_sym(W,num_blocks_wide)
    # Perform the matrix multiplication on the GPU
    Q = size(W, 1)
    n_occ = size(W, 2)
    p = size(W, 3)    
    K = AMDGPU.zeros(Float64, p,p)
    


    # hipstream =  AMDGPU.HIPStream(:high)
    # AMDGPU.stream!(hipstream)

    K_block_width = div(p, num_blocks_wide)
    lower_triangle_length = get_triangle_matrix_length(num_blocks_wide)
    exchange_batch_indexes = Array{Tuple{Int, Int}, 1}(undef, lower_triangle_length)
    the_batch_index = 1
    for iii in 1:num_blocks_wide
        for jjj in 1:iii
            exchange_batch_indexes[the_batch_index] = (iii, jjj)
            the_batch_index+=1
        end
    end


    k_blocks = AMDGPU.zeros(Float64, K_block_width, K_block_width, lower_triangle_length)
    p_remainder = p % K_block_width  
    k_last_row_blocks = AMDGPU.zeros(Float64, K_block_width + p_remainder, K_block_width, num_blocks_wide-1)
    k_last_block = AMDGPU.zeros(Float64, K_block_width + p_remainder, K_block_width + p_remainder)

    A_views = Array{ROCArray}(undef, lower_triangle_length)
    B_views = Array{ROCArray}(undef, lower_triangle_length)
    C_views = Array{ROCArray}(undef, lower_triangle_length)
    p_ranges = Array{UnitRange{Int64}}(undef, lower_triangle_length)
    q_ranges = Array{UnitRange{Int64}}(undef, lower_triangle_length)
    M_array = Array{Int64}(undef, lower_triangle_length)
    N_array = Array{Int64}(undef, lower_triangle_length)

    transA = 'T'
    transB = 'N'
    alpha = -1.0
    beta = 1.0




    gemm_time = 0.0
    K_gemm = Q*n_occ
    for block_index in 1:lower_triangle_length
        pp,qq = exchange_batch_indexes[block_index]


        if pp == num_blocks_wide && qq == num_blocks_wide && p % K_block_width != 0
            p_ranges[block_index] = (pp-1)*K_block_width+1:p
            q_ranges[block_index] = (qq-1)*K_block_width+1:p
            M_array[block_index] = K_block_width + p_remainder
            N_array[block_index] = K_block_width + p_remainder

            A_views[block_index] = reshape(view(W, :,:, p_ranges[block_index]), (K_gemm, M_array[block_index]))
            B_views[block_index] = reshape(view(W, :,:, q_ranges[block_index]), (K_gemm, N_array[block_index]))
            C_views[block_index] = k_last_block            
        elseif pp == num_blocks_wide && p % K_block_width != 0
            p_ranges[block_index] = (pp-1)*K_block_width+1:p
            q_ranges[block_index] = (qq-1)*K_block_width+1:qq*K_block_width
            M_array[block_index] = K_block_width + p_remainder
            N_array[block_index] = K_block_width
            A_views[block_index] =  reshape(view(W, :,:, p_ranges[block_index]), (K_gemm, M_array[block_index]))
            B_views[block_index] =  reshape(view(W, :,:, q_ranges[block_index]), (K_gemm, K_block_width))
            C_views[block_index] =  view(k_last_row_blocks, :, :, qq)
        else 
            p_ranges[block_index]  = (pp-1)*K_block_width+1:pp*K_block_width
            q_ranges[block_index]  = (qq-1)*K_block_width+1:qq*K_block_width
            M_array[block_index] = K_block_width
            N_array[block_index] = K_block_width
            A_views[block_index] = reshape(view(W, :,:, p_ranges[block_index]), (K_gemm, K_block_width))
            B_views[block_index] = reshape(view(W, :,:, q_ranges[block_index]), (K_gemm, K_block_width))
            C_views[block_index] = view(k_blocks, :, :, block_index)

        end
    end
    gemm_time = @elapsed begin
        for block_index in 1:lower_triangle_length
            AMDGPU.rocBLAS.gemm!(transA,transB,alpha, A_views[block_index], B_views[block_index],beta, C_views[block_index])
            # AMDGPU.rocBLAS.rocblas_dgemm(hndl, AMDGPU.rocBLAS.rocblas_operation_transpose,
            # AMDGPU.rocBLAS.rocblas_operation_none, M_array[block_index], N_array[block_index], K_gemm, Ref(alpha), 
            # A_views[block_index], K_gemm, B_views[block_index], K_gemm, Ref(beta), C_views[block_index], M_array[block_index])
        end
        AMDGPU.synchronize()
    


    # @time begin 
        for block_index in 1:lower_triangle_length
            copyto!(view(K, p_ranges[block_index], q_ranges[block_index]), C_views[block_index])
            copyto!(view(K, q_ranges[block_index], p_ranges[block_index]), transpose(C_views[block_index]))
        end
        AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    # end
    end
    println("gemm_time: ", gemm_time)


    return K , gemm_time
end


function gpu_gemm_sym_occ_range(W)
    # Perform the matrix multiplication on the GPU
    Q = size(W, 1)
    n_occ = size(W, 2)
    p = size(W, 3)    
    K = AMDGPU.zeros(Float64, p,p)
    num_blocks_wide = 3


    # hipstream =  AMDGPU.HIPStream(:high)
    # AMDGPU.stream!(hipstream)

    K_block_width = div(p, num_blocks_wide)
    lower_triangle_length = get_triangle_matrix_length(num_blocks_wide)
    exchange_batch_indexes = Array{Tuple{Int, Int}, 1}(undef, lower_triangle_length)
    the_batch_index = 1
    for iii in 1:num_blocks_wide
        for jjj in 1:iii
            exchange_batch_indexes[the_batch_index] = (iii, jjj)
            the_batch_index+=1
        end
    end


    k_blocks = AMDGPU.zeros(Float64, K_block_width, K_block_width, lower_triangle_length)
    p_remainder = p % K_block_width  
    k_last_row_blocks = AMDGPU.zeros(Float64, K_block_width + p_remainder, K_block_width, num_blocks_wide-1)
    k_last_block = AMDGPU.zeros(Float64, K_block_width + p_remainder, K_block_width + p_remainder)

    A_views = Array{ROCArray}(undef, lower_triangle_length)
    B_views = Array{ROCArray}(undef, lower_triangle_length)
    C_views = Array{ROCArray}(undef, lower_triangle_length)
    p_ranges = Array{UnitRange{Int64}}(undef, lower_triangle_length)
    q_ranges = Array{UnitRange{Int64}}(undef, lower_triangle_length)
    M_array = Array{Int64}(undef, lower_triangle_length)
    N_array = Array{Int64}(undef, lower_triangle_length)

    transA = 'T'
    transB = 'N'
    alpha = -1.0
    beta = 1.0


    # divide the occ index into 10 ranges
    n_occ_ranges = Array{UnitRange{Int64}}(undef, 10)
    for i in 1:10
        if i == 10
            n_occ_ranges[i] = (i-1)*10+1:n_occ
        else
            n_occ_ranges[i] = (i-1)*10+1:i*10
        end
    end

    println("n_occ_ranges: ", n_occ_ranges)
    gemm_time = 0.0
    for i in 1:10
        occ_range = n_occ_ranges[i]
        K_gemm = Q*length(occ_range)

        for block_index in 1:lower_triangle_length
            pp,qq = exchange_batch_indexes[block_index]
        

            if pp == num_blocks_wide && qq == num_blocks_wide && p % K_block_width != 0
                p_ranges[block_index] = (pp-1)*K_block_width+1:p
                q_ranges[block_index] = (qq-1)*K_block_width+1:p
                M_array[block_index] = K_block_width + p_remainder
                N_array[block_index] = K_block_width + p_remainder

                A_views[block_index] = reshape(view(W, :,occ_range, p_ranges[block_index]), (K_gemm, M_array[block_index]))
                B_views[block_index] = reshape(view(W, :,occ_range, q_ranges[block_index]), (K_gemm, N_array[block_index]))
                C_views[block_index] = k_last_block            
            elseif pp == num_blocks_wide && p % K_block_width != 0
                p_ranges[block_index] = (pp-1)*K_block_width+1:p
                q_ranges[block_index] = (qq-1)*K_block_width+1:qq*K_block_width
                M_array[block_index] = K_block_width + p_remainder
                N_array[block_index] = K_block_width
                A_views[block_index] =  reshape(view(W, :,occ_range, p_ranges[block_index]), (K_gemm, M_array[block_index]))
                B_views[block_index] =  reshape(view(W, :,occ_range, q_ranges[block_index]), (K_gemm, K_block_width))
                C_views[block_index] =  view(k_last_row_blocks, :, :, qq)
            else 
                p_ranges[block_index]  = (pp-1)*K_block_width+1:pp*K_block_width
                q_ranges[block_index]  = (qq-1)*K_block_width+1:qq*K_block_width
                M_array[block_index] = K_block_width
                N_array[block_index] = K_block_width
                A_views[block_index] = reshape(view(W, :,occ_range, p_ranges[block_index]), (K_gemm, K_block_width))
                B_views[block_index] = reshape(view(W, :,occ_range, q_ranges[block_index]), (K_gemm, K_block_width))
                C_views[block_index] = view(k_blocks, :, :, block_index)

            end
        end
        gemm_time += @elapsed begin
            for block_index in 1:lower_triangle_length
                AMDGPU.rocBLAS.gemm!(transA,transB,alpha, A_views[block_index], B_views[block_index],beta, C_views[block_index])
                # AMDGPU.rocBLAS.rocblas_dgemm(hndl, AMDGPU.rocBLAS.rocblas_operation_transpose,
                # AMDGPU.rocBLAS.rocblas_operation_none, M_array[block_index], N_array[block_index], K_gemm, Ref(alpha), 
                # A_views[block_index], K_gemm, B_views[block_index], K_gemm, Ref(beta), C_views[block_index], M_array[block_index])
            end
            AMDGPU.synchronize()
        end
    end

    println("gemm_time: ", gemm_time)

    # @time begin 
        for block_index in 1:lower_triangle_length
            copyto!(view(K, p_ranges[block_index], q_ranges[block_index]), C_views[block_index])
            copyto!(view(K, q_ranges[block_index], p_ranges[block_index]), transpose(C_views[block_index]))
        end
        # AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    # end


    return K, gemm_time
end

function main() 
    # Load the K and W matrices from the HDF5 file
    file_path = "/lustre/orion/chp131/scratch/jahyes1/JuliaChem.jl/JuliaChem-Papers/DF-RHF-Paper/Benchmarks/0.4.3/polyglycine/frontier"
    # K, W = load_K_and_W_hdf5(file_path)

    hipstream =  AMDGPU.HIPStream(:high)
    AMDGPU.stream!(hipstream)
  
    p = 30000
    num_blocks_wide = 4
    Q = p÷num_blocks_wide
    n_occ = 1

    W = AMDGPU.rand(Float64, Q, n_occ, p) # Example random data for W
    # W = AMDGPU.rand(Float64, 2568, 175, 625) # Example random data for W
    K_gpu, no_sym_gemm_time = gpu_gemm_no_sym(W)
    K_symm, sym_gemm_time = gpu_gemm_sym(W, num_blocks_wide)

    K_gpu_host = Array(K_gpu)

    K_symm_host = Array(K_symm)
   min_no_sym_gemm_time = 1e10
   for i in 1:3
       K_gpu, no_sym_gemm_time = gpu_gemm_no_sym(W)
       if no_sym_gemm_time < min_no_sym_gemm_time
           min_no_sym_gemm_time = no_sym_gemm_time
       end
   end
    min_sym_time=  1e10
    for i in 1:3
        K_symm, sym_gemm_time = gpu_gemm_sym(W, num_blocks_wide)
        if sym_gemm_time < min_sym_time
            min_sym_time = sym_gemm_time
        end
    end
 


   

    diff = abs.( K_gpu_host - K_symm_host)
    println("Max difference: ", maximum(diff))
    println("speedup = ", min_no_sym_gemm_time/min_sym_time)

   
end

function do_gemm_batched2(A, C, num_blocks_wide)
    M = size(C, 1)
    N = M
    block_size = div(M, num_blocks_wide)

    K_block_width = div(M, num_blocks_wide)
    lower_triangle_length = get_triangle_matrix_length(num_blocks_wide)
    exchange_batch_indexes = Array{Tuple{Int, Int}, 1}(undef, lower_triangle_length)
    the_batch_index = 1
    for iii in 1:num_blocks_wide
        for jjj in 1:iii
            exchange_batch_indexes[the_batch_index] = (iii, jjj)
            the_batch_index+=1
        end
    end

    blocks = AMDGPU.zeros(Float64, K_block_width, K_block_width, lower_triangle_length)
    #assume M is divisible by num_blocks_wide

    gemm_time = @elapsed begin
        A_views = ROCArray{Float64}(undef, (K_block_width, K_block_width, lower_triangle_length))
        B_views = ROCArray{Float64}(undef, (K_block_width, K_block_width, lower_triangle_length))
        # C_views = Vector{ROCMatrix{Float64}}(undef, (M, M, lower_triangle_length))
        for block_index in 1:lower_triangle_length
            pp,qq = exchange_batch_indexes[block_index]
            p_range = (pp-1)*K_block_width+1:pp*K_block_width
            q_range = (qq-1)*K_block_width+1:qq*K_block_width

            A_views[:,:,block_index] = reshape(view(A, :,p_range), (K_block_width, K_block_width))
            B_views[:,:,block_index] = reshape(view(A, :,q_range), (K_block_width, K_block_width))
            # C_views[:,:,block_index] = view(blocks, :, :, block_index)
            AMDGPU.rocBLAS.gemm!('T','N',-1.0, A_views[:,:,block_index], B_views[:,:,block_index], 0.0, blocks[:,:,block_index])
        end
        
        # reshape_A = reshape(A_views, (M, K_block_width, lower_triangle_length))
        # println("size of A: ", size(A))
        # AMDGPU.rocBLAS.gemm_batched!('T','N',-1.0, A_views, B_views, 0.0, blocks)
        # AMDGPU.synchronize()

        for block_index in 1:lower_triangle_length
            pp,qq = exchange_batch_indexes[block_index]
            p_range = (pp-1)*K_block_width+1:pp*K_block_width
            q_range = (qq-1)*K_block_width+1:qq*K_block_width
            copyto!(view(C, p_range, q_range), view(blocks, :, :, block_index))
            copyto!(view(C, q_range, p_range), transpose(view(C, p_range, q_range)))
        end
        AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    end
    return gemm_time

end

function do_gemm_no_symm(A,C)

    gemm_time = @elapsed begin
        AMDGPU.rocBLAS.gemm!('T','N',-1.0, A, A, 0.0, C)
        AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    end
    return gemm_time
end

function main_square_matrix()
    M = 25000
    N = M
    num_blocks_wide = 2
    K= div(M, num_blocks_wide)

    A = AMDGPU.rand(Float64, K, M)
    C_symm = AMDGPU.zeros(Float64, M, N)
    C_no_symm = AMDGPU.zeros(Float64, M, N)
    
    min_no_symm_time = 1e10
    for i in 1:3
        C_no_symm = AMDGPU.zeros(Float64, M, N)
        no_sym_time = do_gemm_no_symm(A, C_no_symm)
        println("no_sym_time: ", no_sym_time)
        if no_sym_time < min_no_symm_time
            min_no_symm_time = no_sym_time
        end
    end

    sym_time = do_gemm_batched2(A, C_symm, num_blocks_wide)
    min_sym_time = 1e10
    for i in 1:3
        C_symm = AMDGPU.zeros(Float64, M, N)
        sym_time = do_gemm_batched2(A, C_symm, num_blocks_wide)
        println("sym_time: ", sym_time)
        if sym_time < min_sym_time
            min_sym_time = sym_time
        end
    end

    C_symm_host = Array(C_symm)
    C_no_symm_host = Array(C_no_symm)
    diff = abs.(C_symm_host - C_no_symm_host)
    println("Max difference: ", maximum(diff))
    println("Speedup: ", min_no_symm_time/min_sym_time)
end

# main_square_matrix()
# main()

function calculate_W_batches(W)
    p = size(W, 3)
    Q =  size(W, 1)
    n_occ = size(W, 2)
    num_blocks_wide = 3

    K_block_width = div(p, num_blocks_wide) # assume p is divisible by num_blocks_wide

    total_contraction_length = Q*n_occ
    contraction_batch_range_size = K_block_width 
    K = AMDGPU.zeros(Float64, p,p)
    #divide up the reshaped 2d Q*n_occ matrix into batches of size p 
    num_batches = 1 # div(Q*n_occ, contraction_batch_range_size)
    batch_ranges = Array{UnitRange{Int64}}(undef, num_batches)

    for i in 1:num_batches
        if i == num_batches
            batch_ranges[i] = (i-1)*contraction_batch_range_size+1:Q*n_occ
        else
            batch_ranges[i] = (i-1)*contraction_batch_range_size+1:i*contraction_batch_range_size
        end
    end


    #do gemm in lower triangle 
    lower_triangle_length = get_triangle_matrix_length(num_blocks_wide)
    exchange_batch_indexes = Array{Tuple{Int, Int}, 1}(undef, lower_triangle_length)
    the_batch_index = 1
    for iii in 1:num_blocks_wide
        for jjj in 1:iii
            exchange_batch_indexes[the_batch_index] = (iii, jjj)
            the_batch_index+=1
        end
    end

    k_blocks = AMDGPU.zeros(Float64, K_block_width, K_block_width, lower_triangle_length)
    AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete

    #create views of the gemms 
    A_views = ROCArray{Float64}(undef, (total_contraction_length, K_block_width, lower_triangle_length))
    B_views = ROCArray{Float64}(undef, (total_contraction_length, K_block_width, lower_triangle_length))
    C_views = ROCArray{Float64}(undef, (K_block_width, K_block_width, lower_triangle_length))

    p_ranges = Array{UnitRange{Int64}}(undef, lower_triangle_length)
    q_ranges = Array{UnitRange{Int64}}(undef, lower_triangle_length)


    for block_index in 1:lower_triangle_length
        pp,qq = exchange_batch_indexes[block_index]
        p_range = (pp-1)*K_block_width+1:pp*K_block_width
        q_range = (qq-1)*K_block_width+1:qq*K_block_width
        p_ranges[block_index] = p_range
        q_ranges[block_index] = q_range
    end
    gemm_time = @elapsed for batch_index in 1:num_batches
       

        for block_index in 1:lower_triangle_length
            A_views[:,:,block_index] = view(reshape(W, (total_contraction_length, p)), batch_ranges[batch_index], p_ranges[block_index])
            B_views[:,:,block_index] = view(reshape(W, (total_contraction_length, p)), batch_ranges[batch_index], q_ranges[block_index])   
        end


        
        # #do batched gemm 
        AMDGPU.rocBLAS.gemm_batched!('T','N',-1.0, A_views, B_views, 0.0, k_blocks)
        AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
        for block_index in 1:lower_triangle_length
            pp,qq = exchange_batch_indexes[block_index]
         
            view(K, p_ranges[block_index], q_ranges[block_index]) .+= view(k_blocks, :, :, block_index)
            if pp != qq
                view(K, q_ranges[block_index], p_ranges[block_index]) .+= transpose(view(k_blocks, :, :, block_index))
            end
        end
        # AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    end

    return K, gemm_time


end


function calculate_W_batches_low_level(W)

    p = size(W, 3)
    Q =  size(W, 1)
    n_occ = size(W, 2)
    num_blocks_wide = 3

    K_block_width = div(p, num_blocks_wide) # assume p is divisible by num_blocks_wide

    total_contraction_length = Q*n_occ
    contraction_batch_range_size = K_block_width 
    K = AMDGPU.zeros(Float64, p,p)
    #divide up the reshaped 2d Q*n_occ matrix into batches of size p 
    num_batches = div(Q*n_occ, contraction_batch_range_size)
    batch_ranges = Array{UnitRange{Int64}}(undef, num_batches)

    for i in 1:num_batches
        if i == num_batches
            batch_ranges[i] = (i-1)*contraction_batch_range_size+1:Q*n_occ
        else
            batch_ranges[i] = (i-1)*contraction_batch_range_size+1:i*contraction_batch_range_size
        end
    end


    #do gemm in lower triangle 
    lower_triangle_length = get_triangle_matrix_length(num_blocks_wide)
    exchange_batch_indexes = Array{Tuple{Int, Int}, 1}(undef, lower_triangle_length)
    the_batch_index = 1
    for iii in 1:num_blocks_wide
        for jjj in 1:iii
            exchange_batch_indexes[the_batch_index] = (iii, jjj)
            the_batch_index+=1
        end
    end

    k_blocks = AMDGPU.zeros(Float64, K_block_width, K_block_width, lower_triangle_length)
    AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete

    #create views of the gemms 
    A_views = Vector{Ptr{Float64}}(undef, lower_triangle_length)
    B_views = Vector{Ptr{Float64}}(undef, lower_triangle_length)
    C_views = Vector{Ptr{Float64}}(undef, lower_triangle_length)

    p_ranges = Array{UnitRange{Int64}}(undef, lower_triangle_length)
    q_ranges = Array{UnitRange{Int64}}(undef, lower_triangle_length)

    linear_indices_W = LinearIndices(reshape(W, (total_contraction_length, p)))

    for block_index in 1:lower_triangle_length
        pp,qq = exchange_batch_indexes[block_index]
        p_range = (pp-1)*K_block_width+1:pp*K_block_width
        q_range = (qq-1)*K_block_width+1:qq*K_block_width
        p_ranges[block_index] = p_range
        q_ranges[block_index] = q_range
    end
    hndl = AMDGPU.rocBLAS.handle()
    for batch_index in 1:num_batches-1
        for block_index in 1:lower_triangle_length
            reshaped_W = reshape(W, (total_contraction_length, p))
         

            A_views[block_index] = pointer(view(reshaped_W, batch_ranges[batch_index], p_ranges[block_index]),1)
            B_views[block_index] = pointer(view(reshaped_W, batch_ranges[batch_index], q_ranges[block_index]),1)
            C_views[block_index] = pointer(view(k_blocks, :, :, block_index),1)
        end

        #do batched gemm 
        A_array = AMDGPU.ROCArray(A_views)
        B_array = AMDGPU.ROCArray(B_views)
        C_array = AMDGPU.ROCArray(C_views)
        AMDGPU.rocBLAS.rocblas_dgemm_batched(hndl,
            AMDGPU.rocBLAS.rocblas_operation_transpose,
            AMDGPU.rocBLAS.rocblas_operation_none,
            K_block_width, K_block_width, contraction_batch_range_size,
            Ref(-1.0), A_views, contraction_batch_range_size, B_views, contraction_batch_range_size,
            Ref(0.0), C_views, K_block_width, lower_triangle_length)

        AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
        for block_index in 1:lower_triangle_length
            pp,qq = exchange_batch_indexes[block_index]
         
            view(K, p_ranges[block_index], q_ranges[block_index]) .+= view(k_blocks, :, :, block_index)
            if pp != qq
                view(K, q_ranges[block_index], p_ranges[block_index]) .+= transpose(view(k_blocks, :, :, block_index))
            end
        end
        # AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    end

end


function main_calculate_W_batches_vs_no_sym()
    # AMDGPU.device!(2)
    p = 100
    Q = p*4
    n_occ = p ÷ 4 

    W = AMDGPU.zeros(Float64, Q, n_occ, p) # Example random data for W
    for pp in 1:p
        view(W, :, :, pp) .= AMDGPU.rand(Float64, Q, n_occ)
    end
    AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    K_no_symm = AMDGPU.zeros(Float64, p,p)
    K_symm = AMDGPU.zeros(Float64, p,p)

    no_sym_min_time = 1e10
    for i in 1:3
        K_no_symm, no_sym_time = gpu_gemm_no_sym(W)
        println("no_sym_time: ", no_sym_time)
        if no_sym_time < no_sym_min_time
            no_sym_min_time = no_sym_time
        end
    end

    sym_batched_min_time = 1e10
    for i in 1:3
        K_symm, sym_batched_time = calculate_W_batches_low_level(W)
        println("sym_batched_time: ", sym_batched_time)
        if sym_batched_time < sym_batched_min_time
            sym_batched_min_time = sym_batched_time
        end
    end
    k_no_sym_cpu = Array(K_no_symm)
    k_sym_cpu = Array(K_symm)


    
    diff = abs.(k_no_sym_cpu - k_sym_cpu)
    ratio =K_no_symm ./ K_symm
    println("Max difference: ", maximum(diff))
    speedup = no_sym_min_time/sym_batched_min_time  
    println("Speedup: ", speedup)

end

# main_calculate_W_batches_vs_no_sym()

function simple_amdg_batched_gemm_example(p,QQ,n_occ,num_blocks_wide)
    # Example matrices
    # p = 1404
    # QQ = 5658
    # n_occ = 225
    # p = 1000
    # QQ = 4113
    # n_occ = 200
    
    # p = 852
    # QQ = 3495
    # n_occ = 340÷2
    # num_blocks_wide = 2
    

    block_width = p÷num_blocks_wide

    println("block_width: ", block_width)

    Q = n_occ*QQ

    AA = AMDGPU.zeros(Float64, QQ, n_occ, p)

    for pp in 1:p
        view(AA, :, :, pp) .= AMDGPU.rand(Float64, QQ, n_occ)
    end
    AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete

    A= reshape(AA, ( Q, p))
    lower_triangle_length = get_triangle_matrix_length(num_blocks_wide)

    C = AMDGPU.zeros(Float64, block_width, block_width, lower_triangle_length)
    C_dest = AMDGPU.zeros(Float64, p, p)
    lin_indicies_A = LinearIndices(A)
    lin_indicies_C = LinearIndices(C)

    exchange_batch_indexes = Array{Tuple{Int, Int}, 1}(undef, lower_triangle_length)
    the_batch_index = 1
    for iii in 1:num_blocks_wide
        for jjj in 1:iii
            exchange_batch_indexes[the_batch_index] = (iii, jjj)
            the_batch_index+=1
        end
    end

    a_views = Vector{Ptr{Float64}}(undef, lower_triangle_length)
    b_views = Vector{Ptr{Float64}}(undef, lower_triangle_length)
    c_views = Vector{Ptr{Float64}}(undef, lower_triangle_length)
    
    for block_index in 1:lower_triangle_length
        pp,qq = exchange_batch_indexes[block_index]
        p_range = (pp-1)*block_width+1:pp*block_width
        q_range = (qq-1)*block_width+1:qq*block_width
        a_views[block_index] = pointer(view(A, :, p_range,1))
        b_views[block_index] = pointer(view(A, :, q_range,1))
        c_views[block_index] = pointer(view(C, :,:, block_index))
    end
   

    a_pointers = AMDGPU.ROCArray(a_views)
    b_pointers = AMDGPU.ROCArray(b_views)
    c_pointers = AMDGPU.ROCArray(c_views)

  

    hndl = AMDGPU.rocBLAS.handle()

    symm_time = @elapsed begin 
        AMDGPU.rocBLAS.rocblas_dgemm_batched(hndl,
        'T',
        'N',
        block_width, block_width, Q,
        Ref(-1.0), a_pointers, Q, b_pointers, Q,
        Ref(0.0), c_pointers, block_width, lower_triangle_length)

        AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
        for block_index in 1:lower_triangle_length
            pp,qq = exchange_batch_indexes[block_index]
            p_range = (pp-1)*block_width+1:pp*block_width
            q_range = (qq-1)*block_width+1:qq*block_width
            copyto!(view(C_dest, p_range, q_range), view(C, :, :, block_index))
            if pp != qq
                copyto!(view(C_dest, q_range, p_range), transpose(view(C, :, :, block_index)))
            end
        end

        AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    end
    C_simple = AMDGPU.zeros(Float64, p,p)
    AMDGPU.synchronize()
    simple_time = @elapsed begin 

        AMDGPU.rocBLAS.gemm!('T','N',-1.0, A, A, 0.0, C_simple)
        AMDGPU.synchronize() # Synchronize the GPU to ensure all operations are complete
    end

    # println("simple_time: ", simple_time)
    println("symm_time: ", symm_time)

    C_cpu = Array(C_dest)
    C_cpu_simple = Array(C_simple)
    max_diff = maximum(abs.(C_dest - C_simple))  
    diff = abs.(C_cpu - C_cpu_simple)
    println("Max difference: ", max_diff)
    println("speedup: ", simple_time/symm_time)
    # display(diff)
    # display(C_cpu_simple)
    # display(C_cpu)

    if num_blocks_wide == 1
       return 1.0, simple_time
    end

    return  simple_time/symm_time, symm_time
end

function do_stuff()
indexes = [
[100, 405, 40],
[175, 714, 70],
[250, 1023, 100],
[325, 1332, 130],
[400, 1641, 160],
[475, 1950, 190],
[550, 2259, 220],
[625, 2568, 250],
[700, 2877, 280],
[775, 3186, 310],
[850, 3495, 340],
[925, 3804, 370],
[1000, 4113, 400],
[1075, 4422, 430],
[1150, 4731, 460],
[1225, 5040, 490],
[1300, 5349, 520],
[1375, 5658, 550],
[1450, 5967, 580],
[1525, 6276, 610],
]
 
speedups = zeros(Float64, length(indexes),5)
times = zeros(Float64, length(indexes),5)
i = 1
 for ind in indexes
    println("ind: ", ind)
    p, QQ, n_occ = ind
    for num_blocks_wide in 1:5
        println("num_blocks_wide: ", num_blocks_wide)
        speedups[i,num_blocks_wide], times[i,num_blocks_wide]= simple_amdg_batched_gemm_example((p÷num_blocks_wide)*num_blocks_wide,QQ,n_occ÷2,num_blocks_wide)
    end
    i+=1
 end
 display(times)
 display(speedups)
  
end
do_stuff(   )
