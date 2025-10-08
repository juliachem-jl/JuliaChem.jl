
#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>

#include "./jeri-core.hpp"
#include "./jeri-oei.hpp"
#include "./jeri-prop.hpp"
#include "./jeri-tei.hpp"
#include "./jeri-df-tei.hpp"

extern "C" {
  JLCXX_MODULE define_jeri(jlcxx::Module& mod) {
  // -- initialize/finalize functions --//
  mod.method("initialize", &initialize);
  mod.method("finalize", &finalize);

  // //-- shell information --//
  mod.add_type<libint2::Shell>("Shell");
  mod.method("create_shell", &create_shell);
  jlcxx::stl::apply_stl<libint2::Shell>(mod);

  // //-- atom information --//
  mod.add_type<libint2::Atom>("Atom");
  mod.method("create_atom", &create_atom);
  jlcxx::stl::apply_stl<libint2::Atom>(mod);



  // //-- shell pair information --//
  mod.add_type<libint2::ShellPair>("ShellPair");
  // jlcxx::stl::apply_stl<libint2::ShellPair>(mod);
 
  //-- basis set information --//
  mod.add_type<libint2::BasisSet>("BasisSet")
      .constructor<const std::vector<libint2::Atom> &,
                   const std::vector<std::vector<libint2::Shell>> &>()
                   .method("nbf", &libint2::BasisSet::nbf)
                   .method("shell2bf", &libint2::BasisSet::shell2bf);

  mod.method("precompute_shell_pair_data", &precompute_shell_pair_data);

  // -- oei engine information --//
  mod.add_type<libint2::Engine>("LibIntEngine");
  mod.add_type<OEIEngine>("OEIEngine")
    //.constructor<const std::vector<libint2::Atom>&, 
    //  const std::vector<std::vector<libint2::Shell> >& >()
    .constructor<const std::vector<libint2::Atom>&,
      const libint2::BasisSet&, julia_int>() 
    //.method("basis", &OEIEngine::basis)
    .method("compute_overlap_block", &OEIEngine::compute_overlap_block)
    .method("compute_overlap_grad_block", &OEIEngine::compute_overlap_grad_block)
    .method("compute_kinetic_block", &OEIEngine::compute_kinetic_block)
    .method("compute_kinetic_grad_block", &OEIEngine::compute_kinetic_grad_block)
    .method("compute_nuc_attr_block", &OEIEngine::compute_nuc_attr_block)
    .method("compute_nuc_attr_grad_block", &OEIEngine::compute_nuc_attr_grad_block);

  //-- prop engine information --//
  mod.add_type<PropEngine>("PropEngine")
    .constructor<const std::vector<libint2::Atom>&,
      const libint2::BasisSet&>() 
    .method("compute_dipole_block", &PropEngine::compute_dipole_block);

  //-- tei engine information --//
  mod.add_type<TEIEngine>("TEIEngine")
      .constructor<const libint2::BasisSet &,
                   const std::vector<libint2::ShellPair>,
                   const int, const int &>();

  mod.add_type<RHFTEIEngine>("RHFTEIEngine")
      .constructor<const libint2::BasisSet &,
                   const std::vector<libint2::ShellPair> &>()
      .method("compute_eri_block", &RHFTEIEngine::compute_eri_block);

  mod.add_type<DFRHFTEIEngine>("DFRHFTEIEngine")
      .constructor<const libint2::BasisSet &,
                   const libint2::BasisSet &,
                   const std::vector<libint2::ShellPair>,
                   const std::vector<libint2::ShellPair> &>()
      .method("compute_eri_block_df", &DFRHFTEIEngine::compute_eri_block_df)
      .method("compute_two_center_eri_block", &DFRHFTEIEngine::compute_two_center_eri_block);

  }
} 

