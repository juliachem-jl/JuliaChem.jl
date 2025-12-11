#ifndef JERI_DF_TEI_H
#define JERI_DF_TEI_H

#include <libint2.hpp>
#include <jlcxx/jlcxx.hpp>
#include "jeri-tei.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <limits>
#include <vector>

typedef int64_t julia_int;

//------------------------------------------------------------------------//
//-- Two electron integral engine for Density Fitting --------------------//
//-- Extends TEIEngine adding integral support for   ---------------------//
//-- 2E-3C and 2E-2C integrals (E = electrons, C = Center)----------------//
//------------------------------------------------------------------------//

class DFRHFTEIEngine : public TEIEngine
{

public:
  const libint2::BasisSet *m_auxiliary_basis_set;
  const libint2::ShellPair *m_auxillary_shellpair_data;
  libint2::Engine m_two_center_engine;

  //-- ctors and dtors --//
  DFRHFTEIEngine(
      const libint2::BasisSet &t_basis_set,
      const libint2::BasisSet &t_auxiliary_basis_set,
      const std::vector<libint2::ShellPair> &t_shellpair_data,
      const std::vector<libint2::ShellPair> &t_auxillary_shellpair_data)
      : TEIEngine(t_basis_set, t_shellpair_data, (int)std::max(t_basis_set.max_nprim(), t_auxiliary_basis_set.max_nprim()),
                  (int)std::max(t_basis_set.max_l(), t_auxiliary_basis_set.max_l()))
  {
    m_two_center_engine = libint2::Engine(libint2::Operator::coulomb,
                                          t_auxiliary_basis_set.max_nprim(), t_auxiliary_basis_set.max_l(), 0);
    m_auxiliary_basis_set = &t_auxiliary_basis_set;
    m_auxillary_shellpair_data = t_auxillary_shellpair_data.data();
  }

  ~DFRHFTEIEngine(){};

  //-- member functions --//

  //-- compute_eri_block --//
  //-- returns whether the integrals were screened out --//
  inline bool compute_eri_block_df(jlcxx::ArrayRef<double> eri_block, julia_int auxilary_shell_index, 
                                julia_int shell_index_1, julia_int shell_index_2,
                                julia_int copy_size, julia_int memory_skip)
  {
    auto unitShell = libint2::Shell::unit();
    m_coulomb_eng.compute2<libint2::Operator::coulomb,
                           libint2::BraKet::xs_xx, 0>((*m_auxiliary_basis_set)[auxilary_shell_index - 1], unitShell,
                                                      (*m_basis_set)[shell_index_1 - 1], (*m_basis_set)[shell_index_2 - 1]);
    if (m_coulomb_eng.results()[0] != nullptr)
    { 
      for(int i = 0; i < copy_size; i++)
      {
          eri_block.data()[i] = *(m_coulomb_eng.results()[0] + i);
      }
      return false;
    }
    else
    {
      return true;
    }
  }
  //-- compute_eri_block --//
  //-- returns whether the integrals were screened out --//
  inline bool compute_two_center_eri_block(jlcxx::ArrayRef<double> eri_block, 
  julia_int shell_1_index,
  julia_int shell_2_index,
  julia_int shell_1_basis_count,
  julia_int shell_2_basis_count)
  {    
    auto shell2bf = (*m_auxiliary_basis_set).shell2bf();
    int n_basis_functions  =(*m_auxiliary_basis_set).nbf();
    m_two_center_engine.compute((*m_auxiliary_basis_set)[shell_1_index], (*m_auxiliary_basis_set)[shell_2_index]);
    if (m_two_center_engine.results()[0] != nullptr)
    { 
      for(int i = 0; i < shell_1_basis_count* shell_2_basis_count; i++)
      {
          eri_block.data()[i] = *(m_two_center_engine.results()[0] + i);
      }
      return false;
    }
    else
    {
      return true;
    }
  }
};

#endif /* JERI_DF_TEI_H */
