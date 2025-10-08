#Base.include(@__MODULE__,"../basis/BasisStructs.jl")
"""
  module JCInput
The module required for reading in and processing the selected input file.
Import this module into the script when you need to process an input file
(which will be every single calculation).
"""
module JCBasis

using JuliaChem.JCModules
using JuliaChem.JERI

using CxxWrap
using MPI
using Base.Threads
using HDF5
using Printf

Base.include(@__MODULE__, "BasisHelpers.jl")

"""
  run(args::String)
Perform the operations necessary to read in, process, and extract data from the
selected input file.

One input variable is required:
1. args = The name of the input file.

Two variables are output:
1. input_info = Information gathered from the input file.
2. basis = The basis set shells, determined from the input file.

Thus, proper use of the Input.run() function would look like this:

```
input_info, basis = Input.run(args)
```
"""
function run(molecule, model; output="none")
  comm=MPI.COMM_WORLD

  if MPI.Comm_rank(comm) == 0 && output >= 2
    println("--------------------------------------------------------------------------------")
    println("                       ========================================                 ")
    println("                                GENERATING BASIS SET                            ")
    println("                       ========================================                 ")
    println(" ")
  end

  #== initialize variables ==#
  geometry_array::Vector{Float64} = molecule["geometry"]
  symbols::Vector{String} = molecule["symbols"]
  basis::String = model["basis"]
  build_auxillary = haskey(model, "auxiliary_basis")
  auxiliary_basis::String =  build_auxillary ? model["auxiliary_basis"] : ""  
  charge::Int64 = molecule["molecular_charge"]

  num_atoms::Int64 = length(geometry_array)/3
  geometry_array_t::Matrix{Float64} = reshape(geometry_array,(3,num_atoms))
  geometry::Matrix{Float64} = transpose(geometry_array_t)
  geometry .*= 1.0/0.52917724924 #switch from angs to bohr
  
  atomic_number_mapping::Dict{String,Int64} = create_atomic_number_mapping()
  atomic_mass_mapping::Dict{String,Float64} = create_atomic_mass_mapping()
  shell_am_mapping::Dict{String,Int64} = create_shell_am_mapping()

  mol = Molecule([], StdVector{JERI.Atom}())

  if MPI.Comm_rank(comm) == 0 && output >= 2
    println("----------------------------------------          ")
    println("          Printing basis set...                   ")
    println("----------------------------------------          ")
    println()
  end

  #initialize variables and structs for basis set build
  basis_set_shells = Vector{JCModules.Shell}([])
  shells_cxx = StdVector([ StdVector{JERI.Shell}() for i in 1:55 ]) 
  shells_cxx_added = [ false for i in 1:55 ]

  basis_set_nels = -charge 
  basis_set_norb = 0
  pos = 1
  shell_id = 1
  
  #== relocate center of mass of system to origin ==# 
  center_of_mass = Vector{Float64}([0.0, 0.0, 0.0])
  
  atom_centers = Vector{Vector{Float64}}([])
  atomic_masses = Vector{Float64}([])
  for (atom_idx, symbol) in enumerate(symbols) 
    push!(atom_centers, geometry[atom_idx,:])
    push!(atomic_masses, atomic_mass_mapping[symbol])    
   
    center_of_mass .+= atomic_masses[atom_idx] .* atom_centers[atom_idx]
  end
  center_of_mass ./= sum(atomic_masses)

  for icenter in atom_centers
    icenter .-= center_of_mass
  end
  
  #== create basis set ==#
  h5open(joinpath(@__DIR__, "../../records/bsed.h5"),"r") do bsed
    for (atom_idx, symbol) in enumerate(symbols)
      #== initialize variables needed for shell ==#
      atom_center = atom_centers[atom_idx] 
      atomic_number = atomic_number_mapping[symbol]

      #== create atom objects ==#
      push!(mol, JCModules.Atom(atomic_number, symbol, atom_center))
      push!(mol.mol_cxx, JERI.create_atom(atomic_number, atom_center)) 
       
      basis_set_nels += atomic_number

      pos, basis_set_norb, shell_id = add_shells!(bsed, basis_set_shells, shells_cxx, shells_cxx_added, symbol, basis, 
      shell_am_mapping, atom_idx, atomic_number, pos, atom_center, basis_set_norb, shell_id, output)
 
      shells_cxx_added[atomic_number+1] = true 
      #display(shells_cxx)

      if MPI.Comm_rank(comm) == 0 && output >= 2
        println(" ")
      end
    end
  end

  if build_auxillary
    auxiliary_basis_set_shells, auxillary_shells_cxx, auxiliary_basis_set_norb = build_auxillary_basis(auxiliary_basis, symbols, atom_centers, atomic_number_mapping, shell_am_mapping, output, comm)
  end

  if MPI.Comm_rank(comm) == 0 && output >= 2
    println("----------------------------------------          ")
    println("     Printing basis set metadata...               ")
    println("----------------------------------------          ")
    println()
    println("Basis set: $basis")
    println("Number of basis functions: $basis_set_norb")
    if build_auxillary
      println("Auxillary Basis set: $auxiliary_basis")
      println("Number of auxillary basis functions: $auxiliary_basis_set_norb")
    end 

    println("Number of electrons: $basis_set_nels")
  end
  
  #sort!(basis_set_shells, by = x->((x.nbas*x.nprim),x.am))
  #sort!(basis_set_shells, by = x->(x.atomic_number,x.atom_id))
 
  basis_set_cxx = JERI.BasisSet(mol.mol_cxx, shells_cxx)
  basis_set::Basis = Basis(basis_set_shells, basis_set_cxx, 
    StdVector{JERI.ShellPair}(), basis, 
    basis_set_norb, basis_set_nels)                                       

  precompute_shell_pair_data(basis_set.shpdata_cxx, basis_set.basis_cxx)
  return_val = mol, basis_set
  if build_auxillary   
    auxiliary_basis_set_cxx = JERI.BasisSet(mol.mol_cxx, auxillary_shells_cxx)
    auxiliary_basis_set::Basis = Basis(auxiliary_basis_set_shells, auxiliary_basis_set_cxx, 
    StdVector{JERI.ShellPair}(), auxiliary_basis, 
    auxiliary_basis_set_norb, basis_set_nels)   
    
    precompute_shell_pair_data(auxiliary_basis_set.shpdata_cxx, auxiliary_basis_set.basis_cxx)

    return_val = mol, CalculationBasisSets(basis_set, auxiliary_basis_set)
  end

  if MPI.Comm_rank(comm) == 0 && output >= 2
    println(" ")
    println("                       ========================================                 ")
    println("                                       END BASIS                                ")
    println("                       ========================================                 ")
    println("--------------------------------------------------------------------------------")
  end

  return return_val
end

"""
Builds the auxillary basis set
returns 
  1) auxiliary_basis_set_shells: the auxillary basis set shells
  2) auxillary_shells_cxx: the auxillary basis set to pass to libint (c++)
  3) auxiliary_basis_set_norb: the count auxiliary basis set orbitals
"""
function build_auxillary_basis(auxiliary_basis, symbols, atom_centers, atomic_number_mapping, shell_am_mapping, output, comm)
  #initialize variables and structs for auxillary basis set build if building auxillary basis set 
  auxiliary_basis_set_shells = Vector{JCModules.Shell}([])
  auxillary_shells_cxx = StdVector([ StdVector{JERI.Shell}() for i in 1:55 ]) 
  auxillary_shells_cxx_added = [ false for i in 1:55 ]

  auxiliary_basis_set_norb = 0
  auxillary_pos = 1
  auxillary_shell_id = 1

  if MPI.Comm_rank(comm) == 0 && output >= 2
    println("----------------------------------------          ")
    println("       Printing Auxillary basis set...            ")
    println("----------------------------------------          ")
    println()
  end

  h5open(joinpath(@__DIR__, "../../records/auxilliary_bsed.h5"),"r") do aux_bsed
    for (atom_idx, symbol) in enumerate(symbols)
      #== initialize variables needed for shell ==#
      atom_center = atom_centers[atom_idx] 
      atomic_number = atomic_number_mapping[symbol]
      auxillary_pos, auxiliary_basis_set_norb, auxillary_shell_id = add_shells!(aux_bsed, auxiliary_basis_set_shells, auxillary_shells_cxx, auxillary_shells_cxx_added, symbol, auxiliary_basis, 
      shell_am_mapping, atom_idx, atomic_number, auxillary_pos, atom_center, auxiliary_basis_set_norb, auxillary_shell_id, output)  
      auxillary_shells_cxx_added[atomic_number+1] = true 
    
      if MPI.Comm_rank(comm) == 0 && output >= 2
        println(" ")
      end
    end
  end

  return auxiliary_basis_set_shells, auxillary_shells_cxx, auxiliary_basis_set_norb
end

function add_shells!(bsed, basis_set_shells, shells_cxx, shells_cxx_added, symbol, basis, 
  shell_am_mapping, atom_idx, atomic_number, pos, atom_center, basis_set_norb, shell_id, output)
  #== read in basis set values==#
  shells::Dict{String,Any} = read(
    bsed["$symbol/$basis"])
    comm=MPI.COMM_WORLD
  #== process basis set values into shell objects ==#
  if MPI.Comm_rank(comm) == 0 && output >= 2
    println("Atom #$atom_idx ($symbol):") 
    println("  Shell     Ang. Mom.     Prim.       Exp.         Coeff.")
  end

  for shell_num::Int64 in 1:length(shells)
    new_shell_dict::Dict{String,Any} = shells["$shell_num"]

    new_shell_am::Int64 = shell_am_mapping[new_shell_dict["Shell Type"]]
    
    new_shell_exp = new_shell_dict["Exponents"]
    nprim = length(new_shell_exp)
    
    new_shell_coeff = new_shell_dict["Coefficients"]
    ncoeff = length(new_shell_coeff)

    #== if L shell, divide up ==# 
    if new_shell_am == -1
      #== s component ==#
      if MPI.Comm_rank(comm) == 0 && output >= 2
        for iprim in 1:nprim
          @printf("    %d        L (s)          %d     %.6f     %.6f\n", 
            shell_num, iprim, new_shell_exp[iprim], new_shell_coeff[iprim]) 
        end
      end

      new_shell = JCModules.Shell(shell_id, atom_idx, atomic_number, 
        new_shell_exp, new_shell_coeff[1:nprim],
        atom_center, 1, nprim, pos, true)
      push!(basis_set_shells, new_shell)
      #display(new_shell_coeff[:,1])
      if !shells_cxx_added[atomic_number+1]
        push!(shells_cxx[atomic_number+1], JERI.create_shell(0, 
          new_shell_exp, new_shell_coeff[1:nprim], atom_center))
      end

      basis_set_norb += 1 
      shell_id += 1
      pos += new_shell.nbas

      #== p component ==#
      if MPI.Comm_rank(comm) == 0 && output >= 2
        for iprim in 1:nprim
          @printf("    %d        L (p)          %d      %.6f     %.6f\n", 
            shell_num, iprim, new_shell_exp[iprim], 
            new_shell_coeff[nprim+iprim]) 
        end
      end
      
      new_shell = JCModules.Shell(shell_id, atom_idx, atomic_number,
        new_shell_exp, new_shell_coeff[(nprim+1):ncoeff],
        atom_center, 2, nprim, pos, true)

      push!(basis_set_shells,new_shell)
      #display(new_shell_coeff[:,2])
      if !shells_cxx_added[atomic_number+1]
        push!(shells_cxx[atomic_number+1], JERI.create_shell(1, 
          new_shell_exp, new_shell_coeff[(nprim+1):ncoeff], atom_center))
      end

      basis_set_norb += 3 
      shell_id += 1
      pos += new_shell.nbas
    #== otherwise accept shell as is ==#
    else 
      if MPI.Comm_rank(comm) == 0 && output >= 2
        for iprim in 1:nprim
          @printf("    %d          %s            %d      %.6f     %.6f\n", 
            shell_num, new_shell_dict["Shell Type"], iprim, 
            new_shell_exp[iprim], new_shell_coeff[iprim]) 
        end
      end

      new_shell = JCModules.Shell(shell_id, atom_idx, atomic_number,
        new_shell_exp, deepcopy(new_shell_coeff),
        atom_center, new_shell_am, nprim, pos, true)
      push!(basis_set_shells,new_shell)
      if !shells_cxx_added[atomic_number+1]
        push!(shells_cxx[atomic_number+1], JERI.create_shell(new_shell_am-1, 
          vec(new_shell_exp),  vec(new_shell_coeff), vec(atom_center)))
      end 
      basis_set_norb += new_shell.nbas
      shell_id += 1
      pos += new_shell.nbas
    end

    if MPI.Comm_rank(comm) == 0 && output >= 2
      println(" ")
    end
  end
  return  pos, basis_set_norb, shell_id
end

export run

end
