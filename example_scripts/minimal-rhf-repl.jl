#=============================#
#== put needed modules here ==#
#=============================#
import JuliaChem

#================================#
#== JuliaChem execution script ==#
#================================#
function minimal_rhf(input_file)
  try
    #== read in input file ==#
    molecule, driver, model, keywords = JuliaChem.JCInput.run(input_file;       
      output="verbose")       
    
    #== generate basis set ==#
    mol, basis = JuliaChem.JCBasis.run(molecule, model; 
      output="none")          

    #JuliaChem.JCMolecule.run(mol)

    #== perform scf calculation ==#
    if haskey(keywords, "scf")
      rhf_energy = JuliaChem.JCRHF.Energy.run(mol, basis, keywords["scf"];
        output="verbose")
    else
      rhf_energy = JuliaChem.JCRHF.Energy.run(mol, basis;
        output="verbose")
    end

    #display(rhf_energy["Density"]); println()
    #display(rhf_energy["Energy-Weighted Density"]); println()

    #== perform gradient ==#
    #rhf_gradient = JuliaChem.JCGrad.run(mol, basis; output="verbose")

    return rhf_energy
  catch e                                                                       
    bt = catch_backtrace()                                                      
    msg = sprint(showerror, e, bt)                                              
    println(msg)                                                                
                                                                                
    JuliaChem.finalize()                                                        
    exit()                                                                      
  end   
end
