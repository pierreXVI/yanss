#include "spatial.h"
#include "input.h"
#include "private_impl.h"


/*
  Fills the IS mesh->perio to represent the mesh periodicity
  disp is the displacement from bc_1 to bc_2
*/
static PetscErrorCode MeshSetPeriodicity(Mesh mesh, PetscInt bc_1, PetscInt bc_2, PetscReal *disp){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  IS             is_face1, is_face2;
  const PetscInt *array_face1, *array_face2;
  PetscInt       dim, ghostStart, ghostEnd, nface1_loc, nface2_loc;
  ierr = DMGetDimension(mesh->dm, &dim);                              CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(mesh->dm, &ghostStart, &ghostEnd); CHKERRQ(ierr);
  ierr = DMGetStratumIS(mesh->dm, "Face Sets", bc_1, &is_face1);      CHKERRQ(ierr);
  ierr = DMGetStratumIS(mesh->dm, "Face Sets", bc_2, &is_face2);      CHKERRQ(ierr);
  ierr = ISGetSize(is_face1, &nface1_loc);                            CHKERRQ(ierr);
  ierr = ISGetSize(is_face2, &nface2_loc);                            CHKERRQ(ierr);
  ierr = ISGetIndices(is_face2, &array_face2);                        CHKERRQ(ierr);
  ierr = ISGetIndices(is_face1, &array_face1);                        CHKERRQ(ierr);

  Vec             facecoord_loc, facecoord_mpi, support_loc, support_mpi, rank_loc, rank_mpi;
  const PetscReal *facecoord_array, *support_array, *rank_array;
  VecScatter      ctx;
  PetscInt        facecoord_start, support_start, nface, rank_start;
  ierr = VecCreate(PetscObjectComm((PetscObject) mesh->dm), &facecoord_mpi); CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) mesh->dm), &support_mpi);   CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) mesh->dm), &rank_mpi);   CHKERRQ(ierr);
  ierr = VecSetType(facecoord_mpi, VECMPI);                                  CHKERRQ(ierr);
  ierr = VecSetType(support_mpi, VECMPI);                                    CHKERRQ(ierr);
  ierr = VecSetType(rank_mpi, VECMPI);                                    CHKERRQ(ierr);
  ierr = VecSetSizes(facecoord_mpi, nface2_loc * dim, PETSC_DECIDE);         CHKERRQ(ierr);
  ierr = VecSetSizes(support_mpi, nface2_loc * 2, PETSC_DECIDE);             CHKERRQ(ierr);
  ierr = VecSetSizes(rank_mpi, nface2_loc, PETSC_DECIDE);             CHKERRQ(ierr);
  ierr = VecSetBlockSize(facecoord_mpi, dim);                                CHKERRQ(ierr);
  ierr = VecSetBlockSize(support_mpi, 2);                                    CHKERRQ(ierr);
  ierr = VecGetSize(facecoord_mpi, &nface);                                  CHKERRQ(ierr);
  nface /= dim;
  ierr = VecGetOwnershipRange(facecoord_mpi, &facecoord_start, PETSC_NULL);  CHKERRQ(ierr);
  facecoord_start /= dim;
  ierr = VecGetOwnershipRange(support_mpi, &support_start, PETSC_NULL);      CHKERRQ(ierr);
  support_start /= 2;
  ierr = VecGetOwnershipRange(rank_mpi, &rank_start, PETSC_NULL);      CHKERRQ(ierr);


  // IS is_master, is_slave;
  // PetscInt indices_master[nface1_loc + nface2_loc], loc_master[nface1_loc + nface2_loc], indices_slave[nface1_loc + nface2_loc];


  for (PetscInt j = 0; j < nface2_loc; j++) {
    PetscInt       loc, support_size;
    PetscReal      c_2[dim], support_real[2];
    const PetscInt *support;

    ierr = DMPlexGetSupportSize(mesh->dm, array_face2[j], &support_size); CHKERRQ(ierr);
    if (support_size != 2) {
      for (PetscInt k = 0; k < dim; k++) {c_2[k] = NAN;}
      support_real[0] = -1;
      support_real[1] = -1;
    } else {
      ierr = DMPlexComputeCellGeometryFVM(mesh->dm, array_face2[j], PETSC_NULL, c_2, PETSC_NULL); CHKERRQ(ierr);
      ierr = DMPlexGetSupport(mesh->dm, array_face2[j], &support);                                CHKERRQ(ierr);
      if (ghostStart <= support[0] && support[0] < ghostEnd) {
        support_real[0] = support[1];
        support_real[1] = support[0];
      } else {
        support_real[0] = support[0];
        support_real[1] = support[1];
      }
    }
    // indices_master[j] = support_real[0];
    loc = j + facecoord_start;
    ierr = VecSetValuesBlocked(facecoord_mpi, 1, &loc, c_2, INSERT_VALUES); CHKERRQ(ierr);
    loc = j + support_start;
    ierr = VecSetValuesBlocked(support_mpi, 1, &loc, support_real, INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValue(support_mpi, rank_start + j, rank, INSERT_VALUES); CHKERRQ(ierr);

  }
  // ierr = ISRestoreIndices(is2_loc, &faces2_loc);                                             CHKERRQ(ierr);
  // ierr = ISDestroy(&is2_loc);                                                                CHKERRQ(ierr);

  ierr = VecAssemblyBegin(facecoord_mpi);                                                    CHKERRQ(ierr);
  ierr = VecAssemblyEnd(facecoord_mpi);                                                      CHKERRQ(ierr);
  ierr = VecScatterCreateToAll(facecoord_mpi, &ctx, &facecoord_loc);                         CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx, facecoord_mpi, facecoord_loc, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx, facecoord_mpi, facecoord_loc, INSERT_VALUES, SCATTER_FORWARD);   CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);                                                            CHKERRQ(ierr);
  ierr = VecDestroy(&facecoord_mpi);                                                         CHKERRQ(ierr);
  ierr = VecGetArrayRead(facecoord_loc, &facecoord_array);                                   CHKERRQ(ierr);

  ierr = VecAssemblyBegin(support_mpi);                                                      CHKERRQ(ierr);
  ierr = VecAssemblyEnd(support_mpi);                                                        CHKERRQ(ierr);
  ierr = VecScatterCreateToAll(support_mpi, &ctx, &support_loc);                             CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx, support_mpi, support_loc, INSERT_VALUES, SCATTER_FORWARD);     CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx, support_mpi, support_loc, INSERT_VALUES, SCATTER_FORWARD);       CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);                                                            CHKERRQ(ierr);
  ierr = VecDestroy(&support_mpi);                                                           CHKERRQ(ierr);
  ierr = VecGetArrayRead(support_loc, &support_array);                                       CHKERRQ(ierr);

  ierr = VecAssemblyBegin(rank_mpi);                                                      CHKERRQ(ierr);
  ierr = VecAssemblyEnd(rank_mpi);                                                        CHKERRQ(ierr);
  ierr = VecScatterCreateToAll(rank_mpi, &ctx, &rank_loc);                             CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx, rank_mpi, rank_loc, INSERT_VALUES, SCATTER_FORWARD);     CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx, rank_mpi, rank_loc, INSERT_VALUES, SCATTER_FORWARD);       CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);                                                            CHKERRQ(ierr);
  ierr = VecDestroy(&rank_mpi);                                                           CHKERRQ(ierr);
  ierr = VecGetArrayRead(rank_loc, &rank_array);                                       CHKERRQ(ierr);

  for (PetscInt i = 0; i < nface1_loc; i++) {
    PetscReal c_1[dim], len;
    PetscBool found = PETSC_FALSE;
    PetscInt support_size;
    ierr = DMPlexGetSupportSize(mesh->dm, array_face1[i], &support_size); CHKERRQ(ierr);
    if (support_size != 2) continue;

    ierr = DMPlexComputeCellGeometryFVM(mesh->dm, array_face1[i], &len, c_1, PETSC_NULL); CHKERRQ(ierr);
    for (PetscInt j = 0; j < nface; j++) {
      PetscReal dist = 0;
      for (PetscInt k = 0; k < dim; k++) dist += PetscSqr(facecoord_array[dim * j + k] - c_1[k] - disp[k]);
      if (PetscSqrtReal(dist) / len < 1E-1) {
        found = PETSC_TRUE;

        PetscInt       master[2], slave[2];
        PetscInt const *support;
        ierr = DMPlexGetSupport(mesh->dm, array_face1[i], &support); CHKERRQ(ierr);
        if (ghostStart <= support[0] && support[0] < ghostEnd) {
          master[0] = support[1];
          slave[1] = support[0];
        } else {
          master[0] = support[0];
          slave[1] = support[1];
        }
        master[1] = support_array[2 * j + 0];
        slave[0] = support_array[2 * j + 1];

        PetscPrintf(PETSC_COMM_SELF, "[%d] %d -> %d (%d) & %d -> %d (%d)\n", rank, master[0], slave[0], rank, master[1], slave[1], rank_array[j]);
        break;
      }
    }

    if (!found) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Cannot find periodic face on boundary %d for face %d on boundary %d with given displacement", bc_2, array_face1[i], bc_1);

  }


  // if (rank) PetscSleep(1);
  //
  // // if (!rank){
  //   PetscInt size, supsize;
  //   ierr = ISGetSize(gathered, &size);                                       CHKERRQ(ierr);
  //   // PetscPrintf(PETSC_COMM_SELF, "Checking boundary %d (size %d)\n", bc_2, size);
  //   // ISView(is_2, PETSC_VIEWER_STDOUT_SELF);
  //   // ISView(gathered, PETSC_VIEWER_STDOUT_SELF);
  //   for (PetscInt i = 0; i < nface2; i++) {
  //     DMPlexGetSupportSize(mesh->dm, faces_2[i], &supsize); CHKERRQ(ierr);
  //     PetscReal c_2[dim];
  //     ierr = DMPlexComputeCellGeometryFVM(mesh->dm, faces_2[i], PETSC_NULL, c_2, PETSC_NULL); CHKERRQ(ierr);
  //     PetscPrintf(PETSC_COMM_SELF, "[%d] cell %d : %d in %f, %f ,support size %d\n", rank, i, faces_2[i], c_2[0], c_2[1], supsize);
  //   }
  //
  //   // PetscPrintf(PETSC_COMM_SELF, "[%d] Checking boundary %d DONE\n", rank, bc_2);
  // // }
  // if (!rank) PetscSleep(1);

  // ierr = VecRestoreArrayRead(facecoord_loc, &facecoord_array); CHKERRQ(ierr);
  // ierr = VecRestoreArrayRead(support_loc, &support_array);     CHKERRQ(ierr);
  // ierr = VecDestroy(&facecoord_loc);                           CHKERRQ(ierr);
  // ierr = VecDestroy(&support_loc);                             CHKERRQ(ierr);
  // ierr = ISRestoreIndices(is1_loc, &faces1_loc);               CHKERRQ(ierr);
  // ierr = ISDestroy(&is1_loc);                                  CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "[%d] DONE\n", rank);
  PetscSleep(1);
  if (!rank) PetscFunctionReturn(1);


  // for (PetscInt i = 0; i < nface1; i++) {
  //   PetscReal c_1[dim], c_2[dim], len;
  //   PetscBool found = PETSC_FALSE;
  //
  //   PetscInt supsize;
  //   DMPlexGetSupportSize(mesh->dm, faces_1[i], &supsize); CHKERRQ(ierr);
  //
  //   ierr = DMPlexComputeCellGeometryFVM(mesh->dm, faces_1[i], &len, c_1, PETSC_NULL);         CHKERRQ(ierr);
  //
  //   if (rank) PetscPrintf(PETSC_COMM_SELF, "[%d] cell %d : %d in %f, %f, supsize %d\n", rank, i, faces_1[i], c_1[0], c_1[1], supsize);
  //
  //   for (PetscInt j = 0; j < nface2; j++) {
  //     if (rank) PetscPrintf(PETSC_COMM_SELF, "[%d]     cell %d : %d\n", rank, j, faces_2[j]);
  //
  //     ierr = DMPlexComputeCellGeometryFVM(mesh->dm, faces_2[j], PETSC_NULL, c_2, PETSC_NULL); CHKERRQ(ierr);
  //
  //     if (rank) PetscPrintf(PETSC_COMM_SELF, "[%d]     cell %d : %d in %f, %f\n", rank, j, faces_2[j], c_2[0], c_2[1]);
  //
  //     PetscReal dist = 0;
  //     for (PetscInt k = 0; k < dim; k++) dist += PetscSqr(c_2[k] - c_1[k] - disp[k]);
  //
  //     if (PetscSqrtReal(dist) / len < 1E-1) {
  //       found = PETSC_TRUE;
  //
  //       PetscInt const *support_1, *support_2;
  //       ierr = DMPlexGetSupport(mesh->dm, faces_1[i], &support_1); CHKERRQ(ierr);
  //       ierr = DMPlexGetSupport(mesh->dm, faces_2[j], &support_2); CHKERRQ(ierr);
  //
  //       if (ghoscCStart <= support_1[0] && support_1[0] < ghostCEnd) {
  //         cells_1[i] = support_1[1];
  //         cells_2[i + nface] = support_1[0];
  //       } else {
  //         cells_1[i] = support_1[0];
  //         cells_2[i + nface] = support_1[1];
  //       }
  //
  //       if (ghoscCStart <= support_2[0] && support_2[0] < ghostCEnd) {
  //         cells_1[i + nface] = support_2[1];
  //         cells_2[i] = support_2[0];
  //       } else {
  //         cells_1[i + nface] = support_2[0];
  //         cells_2[i] = support_2[1];
  //       }
  //       break;
  //     }
  //   }
  //
  //   if (!found) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Cannot find periodic face on boundary %d for face %d on boundary %d with given displacement", bc_2, faces_1[i], bc_1);
  // }
  // PetscPrintf(PETSC_COMM_SELF, "[%d] DONE\n", rank);



  // IS is_list[2];
  //
  // is_list[0] = mesh->perio[0];
  // ierr = ISCreateGeneral(PetscObjectComm((PetscObject) mesh->perio[0]), 2 * nface, cells_1, PETSC_OWN_POINTER, &is_list[1]); CHKERRQ(ierr);
  // ierr = ISConcatenate(PetscObjectComm((PetscObject) mesh->perio[0]), 2, is_list, &(mesh->perio[0]));                        CHKERRQ(ierr);
  // ierr = ISDestroy(&is_list[0]);                                                                                             CHKERRQ(ierr);
  // ierr = ISDestroy(&is_list[1]);                                                                                             CHKERRQ(ierr);
  //
  // is_list[0] = mesh->perio[1];
  // ierr = ISCreateGeneral(PetscObjectComm((PetscObject) mesh->perio[1]), 2 * nface, cells_2, PETSC_OWN_POINTER, &is_list[1]); CHKERRQ(ierr);
  // ierr = ISConcatenate(PetscObjectComm((PetscObject) mesh->perio[1]), 2, is_list, &(mesh->perio[1]));                        CHKERRQ(ierr);
  // ierr = ISDestroy(&is_list[0]);                                                                                             CHKERRQ(ierr);
  // ierr = ISDestroy(&is_list[1]);                                                                                             CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



PetscErrorCode MeshDestroy(Mesh *mesh){
  PetscErrorCode ierr;
  PetscFV        fvm;

  PetscFunctionBeginUser;
  ierr = DMGetField((*mesh)->dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fvm);                                        CHKERRQ(ierr);
  ierr = DMDestroy(&(*mesh)->dm);                                     CHKERRQ(ierr);
  ierr = ISDestroy(&(*mesh)->perio[0]);                               CHKERRQ(ierr);
  ierr = ISDestroy(&(*mesh)->perio[1]);                               CHKERRQ(ierr);
  ierr = PetscFree(*mesh);                                            CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MeshLoadFromFile(MPI_Comm comm, const char *filename, const char *opt_filename, Mesh *mesh){
  PetscErrorCode ierr;
  DM             foo_dm;

  PetscFunctionBeginUser;
  ierr = PetscNew(mesh);                                                          CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, &(*mesh)->dm);          CHKERRQ(ierr);
  ierr = DMViewFromOptions((*mesh)->dm, PETSC_NULL, "-dm_view_orig");             CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency((*mesh)->dm, PETSC_TRUE, PETSC_FALSE);               CHKERRQ(ierr);
  ierr = DMPlexDistribute((*mesh)->dm, 1, PETSC_NULL, &foo_dm);                   CHKERRQ(ierr);
  if (foo_dm) {
    ierr = DMDestroy(&(*mesh)->dm);                                               CHKERRQ(ierr);
    (*mesh)->dm = foo_dm;
  }
  ierr = DMSetFromOptions((*mesh)->dm);                                           CHKERRQ(ierr);
  ierr = DMPlexConstructGhostCells((*mesh)->dm, PETSC_NULL, PETSC_NULL, &foo_dm); CHKERRQ(ierr);
  ierr = DMDestroy(&(*mesh)->dm);                                                 CHKERRQ(ierr);
  (*mesh)->dm = foo_dm;
  ierr = PetscObjectSetName((PetscObject) (*mesh)->dm, "Mesh");                   CHKERRQ(ierr);

  PetscFV fvm;
  ierr = PetscFVCreate(comm, &fvm);                              CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fvm, "FV Model");      CHKERRQ(ierr);
  ierr = DMAddField((*mesh)->dm, PETSC_NULL, (PetscObject) fvm); CHKERRQ(ierr);

  ierr = ISCreateGeneral(comm, 0, PETSC_NULL, PETSC_OWN_POINTER, (*mesh)->perio + 0); CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, 0, PETSC_NULL, PETSC_OWN_POINTER, (*mesh)->perio + 1); CHKERRQ(ierr);

  PetscInt       dim, num;
  IS             values_is;
  const PetscInt *values;
  ierr = DMGetDimension((*mesh)->dm, &dim);                    CHKERRQ(ierr);
  ierr = DMGetLabelSize((*mesh)->dm, "Face Sets", &num);       CHKERRQ(ierr);
  ierr = DMGetLabelIdIS((*mesh)->dm, "Face Sets", &values_is); CHKERRQ(ierr);
  ierr = ISGetIndices(values_is, &values);                     CHKERRQ(ierr);

  PetscBool remove[num];
  for (PetscInt i = 0; i < num; i++) remove[i] = PETSC_FALSE;
  for (PetscInt i = 0; i < num; i++) {
    PetscInt  master, i_master;
    PetscReal *disp;
    ierr = IOLoadPeriodicity(opt_filename, values[i], dim, &master, &disp); CHKERRQ(ierr);
    if (disp) {
      ierr = MeshSetPeriodicity(*mesh, master, values[i], disp); CHKERRQ(ierr);
      ierr = PetscFree(disp);                                    CHKERRQ(ierr);

      ierr = ISLocate(values_is, master, &i_master); CHKERRQ(ierr);
      if (i_master < 0)     SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Cannot find  master boundary %d for periodic boundary %d", master, values[i]);
      if (remove[i])        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Cannot set boundary %d as both master and slave", values[i]);
      if (remove[i_master]) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Cannot set boundary %d as both master and slave", master);
      remove[i] = PETSC_TRUE;
      remove[i_master] = PETSC_TRUE;
    }
  }

  DMLabel label_old, label_new;
  ierr = DMRemoveLabel((*mesh)->dm, "Face Sets", &label_old); CHKERRQ(ierr);
  ierr = DMCreateLabel((*mesh)->dm, "Face Sets");             CHKERRQ(ierr);
  ierr = DMGetLabel((*mesh)->dm, "Face Sets", &label_new);    CHKERRQ(ierr);
  for (PetscInt i = 0; i < num; i++) {
    if (!remove[i]) {
      IS points;
      ierr = DMLabelGetStratumIS(label_old, values[i], &points); CHKERRQ(ierr);
      ierr = DMLabelSetStratumIS(label_new, values[i], points);  CHKERRQ(ierr);
      ierr = ISDestroy(&points);                                 CHKERRQ(ierr);
    }
  }
  ierr = DMLabelDestroy(&label_old);           CHKERRQ(ierr);
  ierr = ISRestoreIndices(values_is, &values); CHKERRQ(ierr);
  ierr = ISDestroy(&values_is);                CHKERRQ(ierr);

  ierr = DMTSSetRHSFunctionLocal((*mesh)->dm, MeshDMTSComputeRHSFunctionFVM, *mesh); CHKERRQ(ierr);

  char      opt[] = "____";
  PetscBool flag;
  PetscInt  numGhostCells;
  ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-dm_view", opt, sizeof(opt), PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscStrcmp(opt, "draw", &flag);                                                         CHKERRQ(ierr);
  if (flag) {ierr = DMPlexHideGhostCells((*mesh)->dm, &numGhostCells);                            CHKERRQ(ierr);}
  ierr = DMViewFromOptions((*mesh)->dm, PETSC_NULL, "-dm_view");                                  CHKERRQ(ierr);
  if (flag) {ierr = DMPlexRestoreGhostCells((*mesh)->dm, numGhostCells);                          CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}


PetscErrorCode MeshDMApplyFunction(DM dm, PetscReal time,
                                 PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*),
                                 void *ctx, Vec x){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMProjectFunction(dm, time, &func, &ctx, INSERT_ALL_VALUES, x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MeshDMCreateGlobalVector(DM dm, Vec *x){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dm, x);                                      CHKERRQ(ierr);
  ierr = VecSetOperation(*x, VECOP_VIEW, (void (*)(void)) MyVecView_Plex); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode VecGetComponentVectors(Vec x, PetscInt *Nc, Vec **comps){
  PetscErrorCode ierr;
  PetscInt       n, start, end;
  PetscFV        fvm;
  DM             dm;

  PetscFunctionBeginUser;
  ierr = VecGetDM(x, &dm);                                   CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &n);                   CHKERRQ(ierr);
  ierr = PetscMalloc1(n, comps);                             CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x, &start, &end);              CHKERRQ(ierr);

  IS  is;
  Vec v;
  for (PetscInt i = 0; i < n; i++) {
    ierr = ISCreateStride(PetscObjectComm((PetscObject) x), (end - start) / n, start + i, n, &is); CHKERRQ(ierr);
    ierr = VecGetSubVector(x, is, &v);                                                             CHKERRQ(ierr);
    ierr = VecDuplicate(v, &(*comps)[i]);                                                          CHKERRQ(ierr);
    ierr = VecCopy(v, (*comps)[i]);                                                                CHKERRQ(ierr);
    ierr = VecRestoreSubVector(x, is, &v);                                                         CHKERRQ(ierr);
    ierr = ISDestroy(&is);                                                                         CHKERRQ(ierr);
  }
  if (Nc) *Nc = n;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroyComponentVectors(Vec x, Vec **fields){
  PetscErrorCode ierr;
  PetscInt       Nc;
  PetscFV        fvm;
  DM             dm;

  PetscFunctionBeginUser;
  ierr = VecGetDM(x, &dm);                                   CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);                  CHKERRQ(ierr);
  for (PetscInt i = 0; i < Nc; i++) {
    ierr = VecDestroy(&(*fields)[i]);                        CHKERRQ(ierr);
  }
  ierr = PetscFree(*fields);                                 CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode VecApplyFunctionComponents(Vec x, Vec *y,
                                          PetscErrorCode (*func)(PetscInt, const PetscScalar*, PetscScalar*, void*),
                                          void *ctx){
  PetscErrorCode ierr;
  PetscInt       Nc, start, end, size;
  PetscFV        fvm;
  DM             dm;

  PetscFunctionBeginUser;
  ierr = VecGetDM(x, &dm);                                   CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);                  CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x, &start, &end);              CHKERRQ(ierr);
  size = (end - start) / Nc;

  ierr = VecCreate(PetscObjectComm((PetscObject) x), y); CHKERRQ(ierr);
  ierr = VecSetSizes(*y, size, PETSC_DECIDE);            CHKERRQ(ierr);
  ierr = VecSetFromOptions(*y);                          CHKERRQ(ierr);

  const PetscScalar *val_x;
  PetscScalar       val_y[size]; // TODO: use PetscMalloc1
  PetscInt          ix[size];

  ierr = VecGetArrayRead(x, &val_x);                 CHKERRQ(ierr);
  for (PetscInt i = 0; i < size; i++) {
    ierr = func(Nc, &val_x[Nc * i], &val_y[i], ctx); CHKERRQ(ierr);
    ix[i] = i + start / Nc;
  }
  ierr = VecRestoreArrayRead(x, &val_x);                   CHKERRQ(ierr);
  ierr = VecSetValues(*y, size, ix, val_y, INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(*y);                             CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*y);                               CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



PetscErrorCode MeshDMTSComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *ctx){
  PetscErrorCode ierr;
  Mesh           mesh = (Mesh) ctx;
  PetscReal      *values, *xI, *xG;
  PetscInt       size, Nc;
  const PetscInt *master, *slave;
  PetscFV        fvm;

  PetscFunctionBeginUser;
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);                  CHKERRQ(ierr);
  ierr = ISGetSize(mesh->perio[0], &size);                   CHKERRQ(ierr);
  ierr = ISGetIndices(mesh->perio[0], &master);              CHKERRQ(ierr);
  ierr = ISGetIndices(mesh->perio[1], &slave);               CHKERRQ(ierr);
  ierr = VecGetArray(locX, &values);                         CHKERRQ(ierr);

  for (PetscInt i = 0; i < size; i++) {
    ierr = DMPlexPointLocalFieldRead(dm, master[i], 0, values, &xI); CHKERRQ(ierr);
    ierr = DMPlexPointLocalFieldRef(dm, slave[i], 0, values, &xG);   CHKERRQ(ierr);
    for (PetscInt j = 0; j < Nc; j++) xG[j] = xI[j];
  }

  ierr = VecRestoreArray(locX, &values);            CHKERRQ(ierr);
  ierr = ISRestoreIndices(mesh->perio[0], &master); CHKERRQ(ierr);
  ierr = ISRestoreIndices(mesh->perio[1], &slave);  CHKERRQ(ierr);

  ierr = DMPlexTSComputeRHSFunctionFVM(dm, time, locX, F, ctx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
