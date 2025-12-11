import Test
import JSON

include("../tools/travis/travis-rhf.jl")
#include("s22_gamess_values.jl")

#== select input files ==#
directory = joinpath(@__DIR__, "../example_inputs/S22/")
inputs = readdir(directory)
inputs .= directory .* inputs

display(inputs)
println("")

test_index_str_start = length(ARGS) > 0 ? ARGS[1] : "1"
test_index_start = parse(Int64, test_index_str_start)

test_index_str_end = length(ARGS) > 1 ? ARGS[2] : "22"
test_index_end = parse(Int64, test_index_str_end)
#== initialize JuliaChem ==#
JuliaChem.initialize()

#== read in GAMESS values for comparison ==#
S22_GAMESS_file = open(joinpath(@__DIR__, "s22_gamess_values.json"))
  S22_GAMESS_string = read(S22_GAMESS_file, String)
close(S22_GAMESS_file)

S22_GAMESS = JSON.parse(S22_GAMESS_string)

#== run S22 calculations ==#
molecules = collect(test_index_start:test_index_end) 

s22_test_results = Dict([]) 
s22_test_results_densityfitting = Dict([]) 


travis_rhf_density_fitting(inputs[1], "cc-pVTZ-JKFIT") #get everything initialized before doing timings

for imol in molecules 
  println("S$(imol) starting") 
 
 
  @time begin
    s22_test_results_densityfitting[imol] = travis_rhf_density_fitting(inputs[imol], "cc-pVTZ-JKFIT")
  end
  println("S$(imol) DF-RHF time ^") 

  @time begin
    s22_test_results[imol] = travis_rhf(inputs[imol])
  end
  println("S$(imol) RHF time ^") 

  
end

#== check energies ==#
Test.@testset "S22 Tests" begin

  Test.@testset "S22 Energy" begin
    for imol in molecules 
      println("SCF Energies RHF,DFRHF $(s22_test_results[imol][:Energy]["Energy"]) $(s22_test_results_densityfitting[imol][:Energy]["Energy"])")
      Test.@test s22_test_results[imol][:Energy]["Energy"] ≈ S22_GAMESS["$imol"]["Energy"]
      Test.@test s22_test_results[imol][:Energy]["Energy"] ≈ s22_test_results_densityfitting[imol][:Energy]["Energy"] atol=.0015
    end
  end

  #== check dipole moments ==#
  Test.@testset "S22 Dipoles" begin
    for imol in molecules 
      if S22_GAMESS["$imol"]["Dipole"] == 1.0E-6
        Test.@test abs(s22_test_results[imol][:Properties]["Dipole"][:moment]) <= S22_GAMESS["$imol"]["Dipole"] #check if approximately zero 
      else
        Test.@test s22_test_results[imol][:Properties]["Dipole"][:moment] ≈ S22_GAMESS["$imol"]["Dipole"] atol=5.0E-5
      end
    end
  end

  #== check HOMO-LUMO gaps ==#
  Test.@testset "S22 HOMO-LUMO Gaps" begin
    for imol in molecules 
      Test.@test s22_test_results[imol][:Properties]["MO Energies"][:homo_lumo] ≈ S22_GAMESS["$imol"]["HOMO-LUMO Gap"] atol=5.0E-4
    end
  end

  #== check Mulliken charges ==#
  Test.@testset "S22 Mulliken Charges" begin
    for imol in molecules 
      Test.@test s22_test_results[imol][:Properties]["Mulliken Population"] ≈ 
        S22_GAMESS["$imol"]["Mulliken Population"] atol=5.0E-6
    end
  end

end


#== finalize JuliaChem ==#
JuliaChem.finalize()       
