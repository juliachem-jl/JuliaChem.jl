using MPI

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
function main()
MPI.Init()

two_center_integrals = zeros(Float64, 3420, 3420)

comm = MPI.COMM_WORLD
if MPI.Comm_rank(comm) == 0
    two_center_integrals .= 1.0
else
    two_center_integrals .= 2.0
end

broadcast_two_center_integrals(two_center_integrals)

if MPI.Comm_rank(comm) == 0
    println(two_center_integrals[1:10, 1:10])
    println(two_center_integrals[end-10:end, end-10:end])
end

#barrier 
MPI.Barrier(comm)

if MPI.Comm_rank(comm) == 1
    println(two_center_integrals[1:10, 1:10])
    println(two_center_integrals[end-10:end, end-10:end])
end

MPI.Finalize()

end

main()