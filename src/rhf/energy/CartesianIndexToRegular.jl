# a = Array{String, 3}(undef, (4,2,2))


# for i in 1:4
#     for j in 1:2
#         for k in 1:2
#             a[i,j,k] = "$i,$j,$k"
#         end 
#     end
# end

# a_vec = vec(a)




# # for index in 1:4*2*2
# #     i,j,k = get_value_from_index(a, index)
    
# #     println("$(a_vec[index]) || $i,$j,$k")
# # end





# # dim = size(a)
# # b = CartesianIndices(dim)

# # for index in b
# #     println("a[$index] = $(a[index])")
# # end

# # b = Array{Float16, 2}(undef,4,5)

# # b[1,3] = 12.32
# # b[3,1] = 23234.32
# # b[3,3] = 12.32

# # if b[1,3] == b[3,1] 
# #     println("1,3")
# # end


# # if b[2,2] != b[3,1] 
# #     println("2,2")
# # end


# # if b[3,3] == b[1,3] 
# #     println("1,3")
# # end
