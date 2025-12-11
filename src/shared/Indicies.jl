# some shared functionality for indexes relating to 4-center-2-electron-integrals


#number of shell indices for 4-center-2-electron-integrals (ij|kl) given the length of a basis set
@inline function get_n_shell_indicies(n_shells)
    return (muladd(n_shells,n_shells,n_shells)*(muladd(n_shells,n_shells,n_shells) + 2)) >> 3
end


@inline function decompose_shell_index_ijkl(ijkl_index)
    bra_pair = decompose(ijkl_index)
    ket_pair = ijkl_index - triangular_index(bra_pair)


    ish = decompose(bra_pair)
    jsh = bra_pair - triangular_index(ish)

    ksh = decompose(ket_pair)
    lsh = ket_pair - triangular_index(ksh)

    return bra_pair, ket_pair, ish,jsh,ksh,lsh
end

@inline function triangular_index(a::Int,b::Int)
    return (muladd(a,a,-a) >> 1) + b
end
  
@inline function triangular_index(a::Int)
    return muladd(a,a,-a) >> 1
end

@inline function decompose(input::Int)
    #return ceil(Int,(-1.0+√(1+8*input))/2.0)
    return Base.fptosi(
        Int, 
        Base.ceil_llvm(
        0.5*( 
            Base.Math.sqrt_llvm(
            Base.sitofp(Float64, muladd(8,input,1))
            ) - 1.0
        )
        )
    )
end

function get_value_from_index(array, index)
    indexes = eachindex(view(array))
    cart_index = indexes[index]
    dimensions = size(array)
    i = cart_index[1]
    j  = cart_index[2]
    k  = cart_index[3]
    # i = ((index-1) % dimensions[1])+1
    # j  = (index-1) ÷ dimensions[2]*dimensions[3]
    # k  = 0
    return i,j,k
end



export get_n_shell_indicies, decompose_shell_index_ijkl, triangular_index, decompose, get_value_from_index


#=
"""
	 index(a::Int64,b::Int64)
Summary
======
Triangular indexing determination.

Arguments
======
a = row index

b = column index
"""
=#
# @inline function triangular_index(a::Int,b::Int)
#     return (muladd(a,a,-a) >> 1) + b
#   end
  
#   @inline function triangular_index(a::Int)
#     return muladd(a,a,-a) >> 1
#   end
  
#   @inline function decompose(input::Int)
#     #return ceil(Int,(-1.0+√(1+8*input))/2.0)
#     return Base.fptosi(
#       Int, 
#       Base.ceil_llvm(
#         0.5*( 
#           Base.Math.sqrt_llvm(
#             Base.sitofp(Float64, muladd(8,input,1))
#           ) - 1.0
#         )
#       )
#     )
#   end