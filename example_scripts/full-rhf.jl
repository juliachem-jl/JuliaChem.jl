
include("full-rhf-repl.jl")
#== initialize JuliaChem ==#
JuliaChem.initialize() 

full_rhf(ARGS[1])
JuliaChem.finalize()   
