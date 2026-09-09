// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifdef KYNEMA_UGF_USES_TIOGA

#include "overset/TiogaRef.h"

#include "tioga.h"

#include <iostream>

namespace tioga_kynema_ugf {

TiogaRef&
TiogaRef::self(TIOGA::tioga* tg)
{
  static bool initialized{false};
  static std::unique_ptr<TiogaRef> tgref;

  if (initialized) {
    if (tg != nullptr)
      throw std::runtime_error(
        "Multiple registration of TIOGA object encountered");
  } else {
    if (tg == nullptr) {
      tgref.reset(new TiogaRef());
    } else {
      tgref.reset(new TiogaRef(tg));
    }
    initialized = true;
  }

  return *tgref;
}

TiogaRef::TiogaRef() : tg_(new TIOGA::tioga()), owned_(true) {}

TiogaRef::TiogaRef(TIOGA::tioga* tg) : tg_(tg), owned_(false) {}

TiogaRef::~TiogaRef()
{
  if (owned_ && (tg_ != nullptr)) {
    delete tg_;
    tg_ = nullptr;
  }
}

void
tioga_set_communicator(MPI_Comm comm, int rank, int size)
{
  TiogaRef::self().get().setCommunicator(comm, rank, size);
}

void
tioga_profile()
{
  TiogaRef::self().get().profile();
}

void
tioga_perform_connectivity()
{
  TiogaRef::self().get().performConnectivity();
}

void
tioga_data_update(int nvar, int row_major)
{
  TiogaRef::self().get().dataUpdate(nvar, row_major);
}

void
tioga_get_donor_count(int meshtag, int* dcount, int* fcount)
{
  TiogaRef::self().get().getDonorCount(meshtag, dcount, fcount);
}

void
tioga_get_donor_info(
  int meshtag, int* receptor_info, int* inode, double* frac, int* dcount)
{
  TiogaRef::self().get().getDonorInfo(
    meshtag, receptor_info, inode, frac, dcount);
}

void
tioga_register_grid_data(
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
  stk::mesh::EntityId* cell_gid,
  stk::mesh::EntityId* node_gid)
{
  TiogaRef::self().get().registerGridData(
    meshtag, num_nodes, xyz, iblank, num_wallbc, num_ovsetbc, wall_ids,
    ovset_ids, num_topologies, num_verts, num_cells, tioga_conn, cell_gid
#ifdef TIOGA_HAS_NODEGID
    ,
    node_gid
#endif
  );
}

void
tioga_set_cell_iblank(int meshtag, int* iblank_cell)
{
  TiogaRef::self().get().set_cell_iblank(meshtag, iblank_cell);
}

void
tioga_set_resolutions(int meshtag, double* node_res, double* cell_res)
{
  TiogaRef::self().get().setResolutions(meshtag, node_res, cell_res);
}

void
tioga_register_solution(int meshtag, double* qsol, int ncomp)
{
#if TIOGA_HAS_NGP_IFACE
  constexpr int row_major = 0;
  TiogaRef::self().get().register_unstructured_solution(
    meshtag, qsol, ncomp, row_major);
#else
  TiogaRef::self().get().registerSolution(meshtag, qsol);
#endif
}

} // namespace tioga_kynema_ugf

#endif
