"""
    reorder_mpi_gathered_matrix(array_to_reorder ::Array{Float64}, rank_indicies ::Vector{Vector{Int64}}, set_data_function, set_temp_function, temp :: Array{Float64})

    reorder a matrix that came from an MPI Gather
    # Arguments
    - `array_to_reorder` ::Array{Float64} - the matrix to reorder
    - `rank_indicies` ::Array{UnitRange{Int64}} - the indicies of the matrix that came from each rank
    - `set_data_function` ::Function - the function to set the data in the matrix
    - `set_temp_function` ::Function - the function to set the data in the temp array
    - `temp` ::Array{Float64} - the temp array to use in reordering 

"""
function reorder_mpi_gathered_matrix(array_to_reorder ::Array{Float64}, rank_indicies, set_data_function::Function, set_temp_function::Function, temp :: Array{Float64})
    reordering_axes_length = size(array_to_reorder)[end]
    index_list = zeros(Int64, 0)
    for rank_index in 1:length(rank_indicies)
        index_list = vcat(index_list, rank_indicies[rank_index])
    end
    
    where_indicies_are_stored = similar(index_list)

    for index in 1:length(index_list)
        where_indicies_are_stored[index_list[index]] = index
    end       

    index_to_fill = 1
    i = index_to_fill
    index_where_temp_goes = index_list[1]

    while i < reordering_axes_length
        set_temp_function(temp, array_to_reorder, index_to_fill) # place data in the temp array that is in the place where data will first be shifted
        index_where_data_is = where_indicies_are_stored[index_to_fill]
        while index_where_temp_goes != index_to_fill 
            set_data_function(array_to_reorder, index_to_fill, array_to_reorder, index_where_data_is) # place data in the correct place
            where_indicies_are_stored[index_to_fill] = index_to_fill # indicate the data is now in the correct place  
            index_to_fill = index_where_data_is
            index_where_data_is = where_indicies_are_stored[index_to_fill]  # find the next place to put data
        end
        set_data_function(array_to_reorder, index_to_fill, temp) # place data in the correct place that was stored in the temp array
        where_indicies_are_stored[index_to_fill] = index_to_fill
        while i < reordering_axes_length && where_indicies_are_stored[i] == i
            i += 1
        end
        index_to_fill = i
        index_where_temp_goes = index_list[index_to_fill]
    end
end


function set_data_2D!(matrix, matrix_index, data, data_index)
    matrix[:, matrix_index] .= data[:, data_index]
end

function set_data_2D!(matrix, matrix_index, data)
    matrix[:, matrix_index] .= data
end

function set_data_3D!(matrix, matrix_index, data, data_index)
    matrix[:,:,matrix_index] .= data[:,:,data_index]
end

function set_data_3D!(matrix, matrix_index, data)
    matrix[:,:,matrix_index] .= data
end

function set_temp_2D!(temp, data, index)   
    temp .= data[:,index]
end

function set_temp_3D!(temp, data, index)   
    temp .=  data[:,:,index]
end
