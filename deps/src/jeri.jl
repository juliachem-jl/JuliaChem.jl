using CxxWrap

module JERI
  using CxxWrap

  @wrapmodule(()->joinpath(@__DIR__,"../libjeri.so"),:define_jeri) 

  function __init__()
    @initcxx
  end

  export initialize, finalize 
  export Atom, create_atom 
  export Shell, create_shell 
  export BasisSet, nbf
  export ShellPair, precompute_shell_pair_data
  
  export OEIEngine 
  export compute_overlap_block, compute_overlap_grad_block
  export compute_kinetic_block, compute_kinetic_grad_block
  export compute_nuc_attr_block, compute_nuc_attr_grad_block
  
  export PropEngine, compute_dipole_block
  
  export TEIEngine, compute_eri_block
  export RHFTEIEngine, compute_eri_block
  export DFRHFTEIEngine, compute_eri_block_df,compute_two_center_eri_block
end
