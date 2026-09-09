// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TIOGAREF_H
#define TIOGAREF_H

#include <cstdint>
#include <mpi.h>
#include <memory>
#include <vector>

namespace TIOGA {
class tioga;
}

namespace tioga_kynema_ugf {

class TiogaOptions;

/** Manager for the TIOGA handle
 *
 *  This class manages access to the global TIOGA instance across multiple
 *  realms as well as across multi-solver interfaces (e.g.,
 * amr-wind/kynema-ugf coupling). This class allows both use-cases: 1.
 * kynema-ugf only run where overset might be coupling within just one realm
 * or multiple realms but owns the TIOGA reference, 2. hybrid solver run where
 * an external code interface owns the TIOGA reference.
 *
 */
class TiogaRef
{
public:
  /** Access the reference object
   *
   *  The first invocation can pass a pointer to a valid TIOGA instance which
   *  makes this a non-owning reference holder. Otherwise it will create a new
   *  instance and take ownership of that reference.
   *
   *  \param tg Pointer to an existing TIOGA instance
   */
  static TiogaRef& self(TIOGA::tioga* tg = nullptr);

  ~TiogaRef();

  TiogaRef(const TiogaRef&) = delete;
  TiogaRef& operator=(const TiogaRef&) = delete;

  inline operator TIOGA::tioga&() { return *tg_; }

  //! Access the underlying TIOGA reference
  inline TIOGA::tioga& get() { return *tg_; }

private:
  TiogaRef();

  TiogaRef(TIOGA::tioga* tg);

  TIOGA::tioga* tg_{nullptr};

  bool owned_{false};
};

// --- TIOGA API shims -------------------------------------------------
// Defined in TiogaRef.C, the only TU that includes tioga.h. This keeps
// tioga.h out of translation units that pull in Kokkos/CUDA headers.
void tioga_set_communicator(MPI_Comm comm, int rank, int size);
void tioga_set_options(const TiogaOptions& opts);
void tioga_profile();
void tioga_perform_connectivity();
void tioga_reduce_fringes();
void tioga_data_update(int nvar, int row_major);
void tioga_get_receptor_info(std::vector<int>& receptors);
void tioga_get_donor_count(int meshtag, int* dcount, int* fcount);
void tioga_get_donor_info(
  int meshtag, int* receptor_info, int* inode, double* frac, int* dcount);
void tioga_register_grid_data(
  int meshtag,
  int num_nodes,
  double* xyz,
  int* iblank,
  int num_wallbc,
  int num_ovsetbc,
  int* wall_ids,
  int* ovset_ids,
  int num_topologies,
  int* num_verts,
  int* num_cells,
  int** tioga_conn,
  std::uint64_t* cell_gid,
  std::uint64_t* node_gid);
void tioga_set_cell_iblank(int meshtag, int* iblank_cell);
void tioga_set_resolutions(int meshtag, double* node_res, double* cell_res);
void tioga_register_solution(int meshtag, double* qsol, int ncomp);

} // namespace tioga_kynema_ugf

#endif /* TIOGAREF_H */
