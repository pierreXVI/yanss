#include "spatial.h"
#include "physics.h"
#include "input.h"
#include "view.h"


PetscErrorCode MeshDestroy(DM *dm){
  PetscErrorCode ierr;
  PetscFV        fvm, fvmGrad;
  DM             dmGrad;
  MeshCtx        ctx;

  PetscFunctionBeginUser;
  ierr = DMGetField(*dm, 0, NULL, (PetscObject*) &fvm);        CHKERRQ(ierr);
  ierr = DMPlexGetDataFVM(*dm, fvm, NULL, NULL, &dmGrad);      CHKERRQ(ierr);
  ierr = DMGetField(dmGrad, 0, NULL, (PetscObject*) &fvmGrad); CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fvmGrad);                             CHKERRQ(ierr);

  ierr = PetscFVDestroy(&fvm);                                 CHKERRQ(ierr);
  ierr = DMGetApplicationContext(*dm, &ctx);                   CHKERRQ(ierr);
  for (PetscInt n = 0; n < ctx->n_perio; n++) {
    ierr = VecDestroy(&ctx->perio[n].buffer);                  CHKERRQ(ierr);
    ierr = ISDestroy(&ctx->perio[n].master);                   CHKERRQ(ierr);
    ierr = ISDestroy(&ctx->perio[n].slave);                    CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->perio);                                CHKERRQ(ierr);
  for (PetscInt n = 0; n < ctx->n_cell; n++) {
    ierr = ISDestroy(&ctx->CellCtx[n].neighborhood);           CHKERRQ(ierr);
    ierr = PetscFree(ctx->CellCtx[n].grad_coeff);              CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->CellCtx);                              CHKERRQ(ierr);
  ierr = PetscFree(ctx);                                       CHKERRQ(ierr);
  ierr = DMDestroy(dm);                                        CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MeshLoadFromFile(MPI_Comm comm, const char *filename, const char *opt_filename, DM *dm){
  PetscErrorCode ierr;
  DM             foo_dm;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-mesh_view_orig");      CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(*dm, PETSC_TRUE, PETSC_FALSE);    CHKERRQ(ierr);

  PetscPartitioner part;
  ierr = DMPlexGetPartitioner(*dm, &part);                     CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);                 CHKERRQ(ierr);

  ierr = DMPlexDistribute(*dm, 2, NULL, &foo_dm);              CHKERRQ(ierr);
  if (foo_dm) {
    ierr = DMDestroy(dm);                                      CHKERRQ(ierr);
    *dm = foo_dm;
  }
  ierr = DMSetFromOptions(*dm);                                CHKERRQ(ierr);
  ierr = DMPlexConstructGhostCells(*dm, NULL, NULL, &foo_dm);  CHKERRQ(ierr);
  ierr = DMDestroy(dm);                                        CHKERRQ(ierr);
  *dm = foo_dm;
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");        CHKERRQ(ierr);

  PetscFV fvm;
  ierr = PetscFVCreate(comm, &fvm);                         CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fvm, "FV Model"); CHKERRQ(ierr);
  ierr = DMAddField(*dm, NULL, (PetscObject) fvm);          CHKERRQ(ierr);

  MeshCtx ctx;
  ierr = PetscNew(&ctx);                    CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, ctx); CHKERRQ(ierr);

  ierr = MeshSetViewer(*dm);                         CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-mesh_view"); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode MeshApplyFunction(DM dm, PetscReal time,
                                 PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*),
                                 void *ctx, Vec x){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMProjectFunction(dm, time, &func, &ctx, INSERT_ALL_VALUES, x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MeshCreateGlobalVector(DM dm, Vec *x){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dm, x);                                    CHKERRQ(ierr);
  ierr = VecSetOperation(*x, VECOP_VIEW, (void (*)(void)) VecView_Mesh); CHKERRQ(ierr);

  PetscFV  fvm;
  PetscInt Nc;
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);            CHKERRQ(ierr);
  ierr = VecSetBlockSize(*x, Nc);                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode VecGetComponentVectors(Vec x, PetscInt *Nc, Vec **comps){
  PetscErrorCode ierr;
  PetscInt       n, loc_size;
  VecType        type;
  PetscFV        fvm;
  DM             dm;

  PetscFunctionBeginUser;
  ierr = VecGetDM(x, &dm);                             CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &n);             CHKERRQ(ierr);
  ierr = PetscMalloc1(n, comps);                       CHKERRQ(ierr);
  ierr = VecGetType(x, &type);                         CHKERRQ(ierr);
  ierr = VecGetLocalSize(x, &loc_size);                CHKERRQ(ierr);
  loc_size /= n;

  for (PetscInt i = 0; i < n; i++) {
    ierr = VecCreate(PetscObjectComm((PetscObject) x), &(*comps)[i]); CHKERRQ(ierr);
    ierr = VecSetSizes((*comps)[i], loc_size, PETSC_DECIDE);          CHKERRQ(ierr);
    ierr = VecSetType((*comps)[i], type);                             CHKERRQ(ierr);
    ierr = VecStrideGather(x, i, (*comps)[i], INSERT_VALUES);         CHKERRQ(ierr);
  }
  if (Nc) *Nc = n;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroyComponentVectors(Vec x, Vec **comps){
  PetscErrorCode ierr;
  PetscInt       Nc;
  PetscFV        fvm;
  DM             dm;

  PetscFunctionBeginUser;
  ierr = VecGetDM(x, &dm);                             CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);            CHKERRQ(ierr);
  for (PetscInt i = 0; i < Nc; i++) {
    ierr = VecDestroy(&(*comps)[i]);                   CHKERRQ(ierr);
  }
  ierr = PetscFree(*comps);                            CHKERRQ(ierr);
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
  ierr = VecGetDM(x, &dm);                             CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);            CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x, &start, &end);        CHKERRQ(ierr);
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


/*
  Fills the periodicity context
  disp is the displacement from bc_1 to bc_2
*/
static PetscErrorCode MeshSetUp_Periodicity_Ctx(DM dm, PetscInt bc_1, PetscInt bc_2, PetscReal *disp, struct PerioCtx *ctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscInt Nc, dim, ghostStart, ghostEnd;
  { // Get problem related values
    PetscFV  fvm;
    ierr = DMGetDimension(dm, &dim);                              CHKERRQ(ierr);
    ierr = DMPlexGetGhostCellStratum(dm, &ghostStart, &ghostEnd); CHKERRQ(ierr);
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);          CHKERRQ(ierr);
    ierr = PetscFVGetNumComponents(fvm, &Nc);                     CHKERRQ(ierr);
  }

  PetscInt       nface1_loc, nface2_loc, nface_loc;
  IS             face1_is, face2_is;
  const PetscInt *face1, *face2;
  { // Get face description
    ierr = DMGetStratumIS(dm, "Face Sets", bc_1, &face1_is); CHKERRQ(ierr);
    ierr = DMGetStratumIS(dm, "Face Sets", bc_2, &face2_is); CHKERRQ(ierr);
    if (!face1_is) {ierr = ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_OWN_POINTER, &face1_is); CHKERRQ(ierr);}
    if (!face2_is) {ierr = ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_OWN_POINTER, &face2_is); CHKERRQ(ierr);}
    ierr = ISGetSize(face1_is, &nface1_loc); CHKERRQ(ierr);
    ierr = ISGetSize(face2_is, &nface2_loc); CHKERRQ(ierr);
    ierr = ISGetIndices(face1_is, &face1);   CHKERRQ(ierr);
    ierr = ISGetIndices(face2_is, &face2);   CHKERRQ(ierr);

    for (PetscInt i = 0; i < nface1_loc; i++) {
      PetscInt support_size;
      ierr = DMPlexGetSupportSize(dm, face1[i], &support_size); CHKERRQ(ierr);
      if (support_size != 2) nface1_loc = i;
    }
    for (PetscInt j = 0; j < nface2_loc; j++) {
      PetscInt support_size;
      ierr = DMPlexGetSupportSize(dm, face2[j], &support_size); CHKERRQ(ierr);
      if (support_size != 2) nface2_loc = j;
    }
    nface_loc = nface1_loc + nface2_loc;
  }

  PetscInt        block_start;
  Vec             coord_vec;
  const PetscReal *coord;
  PetscInt  *master_array, *slave_array;
  { // Get face coordinates and support
    Vec        coord_mpi;
    VecScatter scatter;
    ierr = PetscMalloc1(nface_loc, &master_array);                   CHKERRQ(ierr);
    ierr = PetscMalloc1(nface_loc, &slave_array);                    CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject) dm), &coord_mpi); CHKERRQ(ierr);
    ierr = VecSetType(coord_mpi, VECMPI);                            CHKERRQ(ierr);
    ierr = VecSetSizes(coord_mpi, nface_loc * dim, PETSC_DECIDE);    CHKERRQ(ierr);
    ierr = VecSetBlockSize(coord_mpi, dim);                          CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(coord_mpi, &block_start, NULL);      CHKERRQ(ierr);
    block_start /= dim;
    for (PetscInt i = 0; i < nface1_loc; i++) {
      PetscInt  loc;
      PetscReal val[dim];
      loc = i + block_start;
      ierr = DMPlexComputeCellGeometryFVM(dm, face1[i], NULL, val, NULL); CHKERRQ(ierr);
      ierr = VecSetValuesBlocked(coord_mpi, 1, &loc, val, INSERT_VALUES); CHKERRQ(ierr);

      const PetscInt *support;
      ierr = DMPlexGetSupport(dm, face1[i], &support);                    CHKERRQ(ierr);
      if (ghostStart <= support[0] && support[0] < ghostEnd) {
        master_array[i] = support[1];
        slave_array[i] = support[0];
      } else {
        master_array[i] = support[0];
        slave_array[i] = support[1];
      }
    }
    for (PetscInt j = 0; j < nface2_loc; j++) {
      PetscInt  loc;
      PetscReal val[dim];
      loc = j + block_start + nface1_loc;
      ierr = DMPlexComputeCellGeometryFVM(dm, face2[j], NULL, val, NULL); CHKERRQ(ierr);
      ierr = VecSetValuesBlocked(coord_mpi, 1, &loc, val, INSERT_VALUES); CHKERRQ(ierr);

      const PetscInt *support;
      ierr = DMPlexGetSupport(dm, face2[j], &support);                    CHKERRQ(ierr);
      if (ghostStart <= support[0] && support[0] < ghostEnd) {
        master_array[nface1_loc + j] = support[1];
        slave_array[nface1_loc + j] = support[0];
      } else {
        master_array[nface1_loc + j] = support[0];
        slave_array[nface1_loc + j] = support[1];
      }
    }
    ierr = VecAssemblyBegin(coord_mpi);                                                     CHKERRQ(ierr);
    ierr = VecAssemblyEnd(coord_mpi);                                                       CHKERRQ(ierr);
    ierr = VecScatterCreateToAll(coord_mpi, &scatter, &coord_vec);                          CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter, coord_mpi, coord_vec, INSERT_VALUES, SCATTER_FORWARD);  CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, coord_mpi, coord_vec, INSERT_VALUES, SCATTER_FORWARD);    CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter);                                                     CHKERRQ(ierr);
    ierr = VecDestroy(&coord_mpi);                                                          CHKERRQ(ierr);
    ierr = VecGetArrayRead(coord_vec, &coord);                                              CHKERRQ(ierr);
  }

  PetscInt nface_tot;
  ierr = VecGetSize(coord_vec, &nface_tot); CHKERRQ(ierr);
  nface_tot /= dim;

  PetscInt *localToGlobal;
  ierr = PetscMalloc1(nface_loc, &localToGlobal); CHKERRQ(ierr);

  for (PetscInt i = 0; i < nface1_loc; i++) {
    PetscBool found = PETSC_FALSE;
    PetscReal len;
    ierr = DMPlexComputeCellGeometryFVM(dm, face1[i], &len, NULL, NULL); CHKERRQ(ierr);
    for (PetscInt j = 0; j < nface_tot; j++) {
      PetscReal dist = 0;
      for (PetscInt k = 0; k < dim; k++) dist += PetscSqr(coord[dim * j + k] - coord[dim * (block_start + i) + k] - disp[k]);
      if (PetscSqrtReal(dist) / len < 1E-1) {
        found = PETSC_TRUE;
        localToGlobal[i] = j;
        break;
      }
    }
    if (!found) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Cannot find periodic face on boundary %d for face %d on boundary %d with given displacement", bc_2, face1[i], bc_1);
  }
  for (PetscInt j = 0; j < nface2_loc; j++) {
    PetscBool found = PETSC_FALSE;
    PetscReal len;
    ierr = DMPlexComputeCellGeometryFVM(dm, face2[j], &len, NULL, NULL); CHKERRQ(ierr);
    for (PetscInt i = 0; i < nface_tot; i++) {
      PetscReal dist = 0;
      for (PetscInt k = 0; k < dim; k++) dist += PetscSqr(coord[dim * (block_start + nface1_loc + j) + k] - coord[dim * i + k] - disp[k]);
      if (PetscSqrtReal(dist) / len < 1E-1) {
        found = PETSC_TRUE;
        localToGlobal[nface1_loc + j] = i;
        break;
      }
    }
    if (!found) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Cannot find periodic face on boundary %d for face %d on boundary %d with given displacement", bc_1, face2[j], bc_2);
  }

  ierr = ISRestoreIndices(face1_is, &face1);     CHKERRQ(ierr);
  ierr = ISRestoreIndices(face2_is, &face2);     CHKERRQ(ierr);
  ierr = ISDestroy(&face1_is);                   CHKERRQ(ierr);
  ierr = ISDestroy(&face2_is);                   CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coord_vec, &coord); CHKERRQ(ierr);
  ierr = VecDestroy(&coord_vec);                 CHKERRQ(ierr);

  ISLocalToGlobalMapping mapping;
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nface_loc, master_array, PETSC_OWN_POINTER, &ctx->master);               CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nface_loc, slave_array, PETSC_OWN_POINTER, &ctx->slave);                 CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, Nc, nface_loc, localToGlobal, PETSC_OWN_POINTER, &mapping); CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) dm), &ctx->buffer);                                               CHKERRQ(ierr);
  ierr = VecSetType(ctx->buffer, VECMPI);                                                                          CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->buffer, nface_loc * Nc, PETSC_DECIDE);                                                   CHKERRQ(ierr);
  ierr = VecSetBlockSize(ctx->buffer, Nc);                                                                         CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(ctx->buffer, mapping);                                                         CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&mapping);                                                                  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
  Reads the periodicity from the input file, and construct the `perio` context array
  The periodicity contexts can only be created after some of the physical context is filled, as the number of components is needed
*/
static PetscErrorCode MeshSetUp_Periodicity(DM dm, const char *opt_filename){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscInt       dim, num, num_loc; // Mesh dimension, number of boundaries on mesh and on local process
  IS             bnd_is, bnd_is_loc;
  const PetscInt *bnd, *bnd_loc;
  {
    IS bnd_is_mpi;
    ierr = DMGetDimension(dm, &dim);                                                                 CHKERRQ(ierr);
    ierr = DMGetLabelIdIS(dm, "Face Sets", &bnd_is_loc);                                             CHKERRQ(ierr);
    ierr = ISOnComm(bnd_is_loc, PetscObjectComm((PetscObject) dm), PETSC_USE_POINTER , &bnd_is_mpi); CHKERRQ(ierr);
    ierr = ISAllGather(bnd_is_mpi, &bnd_is);                                                         CHKERRQ(ierr);
    ierr = ISDestroy(&bnd_is_mpi);                                                                   CHKERRQ(ierr);
    ierr = ISSortRemoveDups(bnd_is);                                                                 CHKERRQ(ierr);
    ierr = ISGetSize(bnd_is, &num);                                                                  CHKERRQ(ierr);
    ierr = ISGetSize(bnd_is_loc, &num_loc);                                                          CHKERRQ(ierr);
    ierr = ISGetIndices(bnd_is_loc, &bnd_loc);                                                       CHKERRQ(ierr);
    ierr = ISGetIndices(bnd_is, &bnd);                                                               CHKERRQ(ierr);
  }

  MeshCtx ctx;
  ierr = DMGetApplicationContext(dm, &ctx); CHKERRQ(ierr);

  PetscBool       rem[num_loc]; // Boundaries to remove
  struct PerioCtx ctxs[num / 2]; // Periodicity contexts
  {
    ctx->n_perio = 0;
    for (PetscInt i = 0; i < num_loc; i++) rem[i] = PETSC_FALSE;
    for (PetscInt i = 0; i < num; i++) {
      PetscInt master;
      PetscReal *disp;
      ierr = IOLoadPeriodicity(opt_filename, bnd[i], dim, &master, &disp); CHKERRQ(ierr);
      if (disp) {
        PetscInt i_master, i_slave;
        ierr = ISLocate(bnd_is_loc, master, &i_master); CHKERRQ(ierr);
        ierr = ISLocate(bnd_is_loc, bnd[i], &i_slave);  CHKERRQ(ierr);
        if (i_master >= 0) rem[i_master] = PETSC_TRUE;
        if (i_slave >= 0) rem[i_slave] = PETSC_TRUE;
        ierr = MeshSetUp_Periodicity_Ctx(dm, master, bnd[i], disp, &ctxs[ctx->n_perio++]); CHKERRQ(ierr);
        ierr = PetscFree(disp);                                                            CHKERRQ(ierr);
      }
    }
  }

  { // Removing periodic boundaries from "Face Sets"
    DMLabel label_old, label_new;
    ierr = DMRemoveLabel(dm, "Face Sets", &label_old); CHKERRQ(ierr);
    ierr = DMCreateLabel(dm, "Face Sets");             CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "Face Sets", &label_new);    CHKERRQ(ierr);
    for (PetscInt i = 0; i < num_loc; i++) {
      if (!rem[i]) {
        IS points;
        ierr = DMLabelGetStratumIS(label_old, bnd_loc[i], &points); CHKERRQ(ierr);
        ierr = DMLabelSetStratumIS(label_new, bnd_loc[i], points);  CHKERRQ(ierr);
        ierr = ISDestroy(&points);                                  CHKERRQ(ierr);
      }
    }
    ierr = DMLabelDestroy(&label_old); CHKERRQ(ierr);
  }

  { // Cleanup
    ierr = ISRestoreIndices(bnd_is_loc, &bnd_loc); CHKERRQ(ierr);
    ierr = ISRestoreIndices(bnd_is, &bnd);         CHKERRQ(ierr);
    ierr = ISDestroy(&bnd_is_loc);                 CHKERRQ(ierr);
    ierr = ISDestroy(&bnd_is);                     CHKERRQ(ierr);
  }

  { // Setting periodicity context for the mesh
    ierr = PetscMalloc1(ctx->n_perio, &ctx->perio); CHKERRQ(ierr);
    for (PetscInt n = 0; n < ctx->n_perio; n++) ctx->perio[n] = ctxs[n];
  }

  PetscFunctionReturn(0);
}


/*
  Compute geometric factors for gradient reconstruction, stored in the mesh context
*/
static PetscErrorCode MeshSetUp_Gradient(DM dm){
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscFV         fvm;
  Vec             cellgeom;
  MeshCtx         ctx;
  DM              dmCell;
  PetscReal       *dx;
  const PetscReal *cellgeom_a;
  PetscInt        cStart, cStartBoundary, *neighbors, dim;
  { // Getting mesh data
    PetscInt ghost, cStartGhost, maxNumNeighbors;
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);         CHKERRQ(ierr);
    ierr = DMPlexGetDataFVM(dm, fvm, &cellgeom, NULL, NULL);     CHKERRQ(ierr);
    ierr = VecGetDM(cellgeom, &dmCell);                          CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom, &cellgeom_a);               CHKERRQ(ierr);
    ierr = DMPlexGetMaxSizes(dm, &maxNumNeighbors, NULL);        CHKERRQ(ierr);
    maxNumNeighbors = PetscSqr(maxNumNeighbors);
    ierr = PetscFVLeastSquaresSetMaxFaces(fvm, maxNumNeighbors); CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);                             CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);         CHKERRQ(ierr);
    ierr = DMPlexGetGhostCellStratum(dm, &cStartBoundary, NULL); CHKERRQ(ierr);
    for (cStartGhost = cStart; cStartGhost < cStartBoundary; cStartGhost++) {
      ierr = DMGetLabelValue(dm, "ghost", cStartGhost, &ghost);  CHKERRQ(ierr);
      if (ghost >= 0) break;
    }

    ierr = DMGetApplicationContext(dm, &ctx);                                     CHKERRQ(ierr);
    ctx->n_cell = cStartGhost - cStart;
    ierr = PetscMalloc1(ctx->n_cell, &ctx->CellCtx);                              CHKERRQ(ierr);
    ierr = PetscMalloc2(maxNumNeighbors, &neighbors, maxNumNeighbors * dim, &dx); CHKERRQ(ierr);
  }

  for (PetscInt n = 0; n < ctx->n_cell; n++) { // Getting neighbors
    PetscInt       nface, nface1, c = cStart + n, nc, size, numNeighbors = 0;
    const PetscInt *faces, *faces1, *fcells;
    ierr = DMPlexGetConeSize(dm, c, &nface); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &faces);     CHKERRQ(ierr);
    for (PetscInt f = 0; f < nface; f++) {
      ierr = DMPlexGetSupport(dm, faces[f], &fcells); CHKERRQ(ierr);
      nc = (c == fcells[0]) ? fcells[1] : fcells[0];
      if (nc >= cStartBoundary) continue;
      neighbors[numNeighbors++] = nc;
      ierr = DMPlexGetConeSize(dm, nc, &nface1); CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, nc, &faces1);     CHKERRQ(ierr);
      for (PetscInt f1 = 0; f1 < nface1; f1++) {
        ierr = DMPlexGetSupportSize(dm, faces1[f1], &size); CHKERRQ(ierr);
        if (size != 2) continue;
        ierr = DMPlexGetSupport(dm, faces1[f1], &fcells); CHKERRQ(ierr);
        if (fcells[0] != c && fcells[0] != nc && fcells[0] < cStartBoundary) neighbors[numNeighbors++] = fcells[0];
        if (fcells[1] != c && fcells[1] != nc && fcells[1] < cStartBoundary) neighbors[numNeighbors++] = fcells[1];
      }
    }

    ierr = ISCreateGeneral(PETSC_COMM_SELF, numNeighbors, neighbors, PETSC_COPY_VALUES, &ctx->CellCtx[n].neighborhood); CHKERRQ(ierr);
    ierr = ISSortRemoveDups(ctx->CellCtx[n].neighborhood);                                                              CHKERRQ(ierr);
    ierr = PetscMalloc1(numNeighbors * dim, &ctx->CellCtx[n].grad_coeff);                                               CHKERRQ(ierr);
  }

  for (PetscInt n = 0; n < ctx->n_cell; n++) { // Computing gradient coefficients
    PetscInt        numNeighbors;
    const PetscInt  *neighborhood;
    PetscFVCellGeom *cg, *ncg;

    ierr = ISGetSize(ctx->CellCtx[n].neighborhood, &numNeighbors);    CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->CellCtx[n].neighborhood, &neighborhood); CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cStart + n, cellgeom_a, &cg); CHKERRQ(ierr);

    for (PetscInt i = 0; i < numNeighbors; i++){
      ierr = DMPlexPointLocalRead(dmCell, neighborhood[i], cellgeom_a, &ncg); CHKERRQ(ierr);
      for (PetscInt j = 0; j < dim; j++) dx[i * dim + j] = ncg->centroid[j] - cg->centroid[j];
    }
    ierr = PetscFVComputeGradient(fvm, numNeighbors, dx, ctx->CellCtx[n].grad_coeff); CHKERRQ(ierr);
  }

  { // Cleanup
    ierr = VecRestoreArrayRead(cellgeom, &cellgeom_a); CHKERRQ(ierr);
    ierr = PetscFree2(neighbors, dx);                  CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


/*
  Compute the RHS
*/
static PetscErrorCode MeshComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *ctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MeshInsertPeriodicValues(dm, locX); CHKERRQ(ierr);

  Vec locF;
  ierr = DMGetLocalVector(dm, &locF); CHKERRQ(ierr);
  ierr = VecZeroEntries(locF);        CHKERRQ(ierr);

  PetscInt cStart, cEnd, fStart, fEnd;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);

  PetscInt Nc;
  PetscFV  fvm;
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc); CHKERRQ(ierr);

  DM  dmGrad;
  Vec locGrad, cellgeom, facegeom;
  { // Computes the gradient, fills Neumann boundaries
    Vec grad;
    ierr = DMPlexGetDataFVM(dm, fvm, &cellgeom, &facegeom, &dmGrad);                                   CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dmGrad, &grad);                                                           CHKERRQ(ierr);
    ierr = MeshReconstructGradientsFVM(dm, locX, grad);                                                CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmGrad, &locGrad);                                                         CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad);                                 CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad);                                   CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmGrad, &grad);                                                       CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, time, facegeom, cellgeom, locGrad);       CHKERRQ(ierr);
  }

  PetscFVFaceGeom *fgeom;
  PetscReal       *vol;
  PetscScalar     *uL, *uR, *fluxL, *fluxR;
  { // Extracting fields, Riemann solve
    PetscDS  prob;
    PetscInt numFaces;
    ierr = DMGetDS(dm, &prob);                                                                                  CHKERRQ(ierr);
    ierr = DMPlexGetFaceFields(dm, fStart, fEnd, locX, NULL, facegeom, cellgeom, locGrad, &numFaces, &uL, &uR); CHKERRQ(ierr);
    ierr = DMPlexGetFaceGeometry(dm, fStart, fEnd, facegeom, cellgeom, &numFaces, &fgeom, &vol);                CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, numFaces * Nc, MPIU_SCALAR, &fluxL);                                              CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, numFaces * Nc, MPIU_SCALAR, &fluxR);                                              CHKERRQ(ierr);
    ierr = PetscArrayzero(fluxL, numFaces * Nc);                                                                CHKERRQ(ierr);
    ierr = PetscArrayzero(fluxR, numFaces * Nc);                                                                CHKERRQ(ierr);
    ierr = PetscFVIntegrateRHSFunction(fvm, prob, 0, numFaces, fgeom, vol, uL, uR, fluxL, fluxR);               CHKERRQ(ierr);
  }

  { // Filling flux vector
    PetscScalar *fa;
    ierr = VecGetArray(locF, &fa); CHKERRQ(ierr);

    for (PetscInt f = fStart, iface = 0; f < fEnd; f++) {
      const PetscInt *cells;
      PetscInt       ghost;
      PetscScalar    *fL = NULL, *fR = NULL;

      ierr = DMGetLabelValue(dm, "ghost", f, &ghost); CHKERRQ(ierr);
      if (ghost >= 0) continue;

      ierr = DMPlexGetSupport(dm, f, &cells);                                    CHKERRQ(ierr);
      ierr = DMGetLabelValue(dm, "ghost", cells[0], &ghost);                     CHKERRQ(ierr);
      if (ghost < 0) {ierr = DMPlexPointLocalFieldRef(dm, cells[0], 0, fa, &fL); CHKERRQ(ierr);}
      ierr = DMGetLabelValue(dm, "ghost", cells[1], &ghost);                     CHKERRQ(ierr);
      if (ghost < 0) {ierr = DMPlexPointLocalFieldRef(dm, cells[1], 0, fa, &fR); CHKERRQ(ierr);}
      for (PetscInt i = 0; i < Nc; i++) {
        if (fL) fL[i] -= fluxL[iface * Nc + i];
        if (fR) fR[i] += fluxR[iface * Nc + i];
      }
      iface++;
    }
    ierr = VecRestoreArray(locF, &fa); CHKERRQ(ierr);
  }

  { // Cleanup
    ierr = DMPlexRestoreFaceFields(dm, fStart, fEnd, locX, NULL, facegeom, cellgeom, locGrad, NULL, &uL, &uR); CHKERRQ(ierr);
    ierr = DMPlexRestoreFaceGeometry(dm, fStart, fEnd, facegeom, cellgeom, NULL, &fgeom, &vol);                CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, 0, MPIU_SCALAR, &fluxL);                                                     CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, 0, MPIU_SCALAR, &fluxR);                                                     CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmGrad, &locGrad);                                                             CHKERRQ(ierr);

    ierr = DMLocalToGlobalBegin(dm, locF, ADD_VALUES, F); CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, locF, ADD_VALUES, F);   CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locF);               CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode MeshSetUp(DM dm, Physics phys, const char *filename){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = DMTSSetRHSFunctionLocal(dm, MeshComputeRHSFunctionFVM, NULL); CHKERRQ(ierr);

  PetscFV fvm;
  { // Setting PetscFV
    PetscLimiter limiter;
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);   CHKERRQ(ierr);
    ierr = PetscFVSetType(fvm, PETSCFVLEASTSQUARES);       CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvm, phys->dim);     CHKERRQ(ierr);
    ierr = PetscFVGetLimiter(fvm, &limiter);               CHKERRQ(ierr);
    ierr = PetscLimiterSetType(limiter, PETSCLIMITERNONE); CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(fvm);                     CHKERRQ(ierr);
  }

  { // Setting gradient
    PetscFV  fvmGrad;
    char     buffer[64];
    ierr = PetscFVCreate(PetscObjectComm((PetscObject) dm), &fvmGrad); CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvmGrad, phys->dim);             CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvmGrad, phys->dim * phys->dof);    CHKERRQ(ierr);
    for (PetscInt i = 0; i < phys->dof; i++){
      const char *cname;
      ierr = PetscFVGetComponentName(fvm, i, &cname); CHKERRQ(ierr);
      for (PetscInt k = 0; k < phys->dim; k++) {
        ierr = PetscSNPrintf(buffer, sizeof(buffer),"d_%d %s", k, cname);   CHKERRQ(ierr);
        ierr = PetscFVSetComponentName(fvmGrad, phys->dim * i + k, buffer); CHKERRQ(ierr);
      }
    }

    ierr = MeshSetUp_Gradient(dm); CHKERRQ(ierr);

    DM           dmGrad;
    PetscInt     cStart, cEnd;
    PetscSection sectionGrad;
    ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_dmgrad_fvm", (PetscObject*) &dmGrad);                     CHKERRQ(ierr);
    ierr = DMDestroy(&dmGrad);                                                                                  CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);                                                       CHKERRQ(ierr);
    ierr = DMClone(dm, &dmGrad);                                                                                CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &sectionGrad);                                 CHKERRQ(ierr);
    ierr = PetscSectionSetChart(sectionGrad, cStart, cEnd);                                                     CHKERRQ(ierr);
    for (PetscInt c = cStart; c < cEnd; c++) {ierr = PetscSectionSetDof(sectionGrad, c, phys->dim * phys->dof); CHKERRQ(ierr);}
    ierr = PetscSectionSetUp(sectionGrad);                                                                      CHKERRQ(ierr);
    ierr = DMSetLocalSection(dmGrad, sectionGrad);                                                              CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&sectionGrad);                                                                   CHKERRQ(ierr);
    ierr = DMAddField(dmGrad, NULL, (PetscObject) fvmGrad);                                                     CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dm, "DMPlex_dmgrad_fvm", (PetscObject) dmGrad);                     CHKERRQ(ierr);
    ierr = DMDestroy(&dmGrad);                                                                                  CHKERRQ(ierr);
  }

  ierr = MeshSetUp_Periodicity(dm, filename); CHKERRQ(ierr);

  { // Setting boundaries
    PetscDS        prob;
    IS             is;
    const PetscInt *indices;
    ierr = DMCreateDS(dm);                                         CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);                                     CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, 0, phys->riemann_solver); CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 0, phys);                       CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);                                     CHKERRQ(ierr);
    { // Getting boundary ids
      IS bnd_is_mpi, bnd_is_loc;
      ierr = DMGetLabelIdIS(dm, "Face Sets", &bnd_is_loc);                                             CHKERRQ(ierr);
      ierr = ISOnComm(bnd_is_loc, PetscObjectComm((PetscObject) dm), PETSC_USE_POINTER , &bnd_is_mpi); CHKERRQ(ierr);
      ierr = ISAllGather(bnd_is_mpi, &is);                                                             CHKERRQ(ierr);
      ierr = ISDestroy(&bnd_is_loc);                                                                   CHKERRQ(ierr);
      ierr = ISDestroy(&bnd_is_mpi);                                                                   CHKERRQ(ierr);
      ierr = ISSortRemoveDups(is);                                                                     CHKERRQ(ierr);
      ierr = ISGetSize(is, &phys->nbc);                                                                CHKERRQ(ierr);
      ierr = ISGetIndices(is, &indices);                                                               CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(phys->nbc, &phys->bc_ctx);                 CHKERRQ(ierr);

    PetscFunctionList bcList;
    ierr = BCRegister(&bcList); CHKERRQ(ierr);

    for (PetscInt i = 0; i < phys->nbc; i++) {
      void (*bcFunc)(void);

      phys->bc_ctx[i].phys = phys;
      ierr = IOLoadBC(filename, indices[i], phys->dim, phys->bc_ctx + i);  CHKERRQ(ierr);
      ierr = PetscFunctionListFind(bcList, phys->bc_ctx[i].type, &bcFunc); CHKERRQ(ierr);

      if (!strcmp(phys->bc_ctx[i].type, "BC_DIRICHLET")) {
        PrimitiveToConservative(phys, phys->bc_ctx[i].val, phys->bc_ctx[i].val);
      }

      if (!bcFunc) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Unknown boundary condition (%s)", phys->bc_ctx[i].type);
      ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, phys->bc_ctx[i].name, "Face Sets", 0, 0,
                                NULL, bcFunc, NULL, 1, indices + i, phys->bc_ctx + i); CHKERRQ(ierr);
    }
    ierr = PetscFunctionListDestroy(&bcList); CHKERRQ(ierr);
    ierr = ISRestoreIndices(is, &indices);    CHKERRQ(ierr);
    ierr = ISDestroy(&is);                    CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);       CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode MeshInsertPeriodicValues(DM dm, Vec locX){
  PetscErrorCode ierr;
  PetscInt       Nc;
  PetscReal      *locX_array;
  PetscFV        fvm;
  MeshCtx        ctx;

  PetscFunctionBeginUser;
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);            CHKERRQ(ierr);

  ierr = DMGetApplicationContext(dm, &ctx); CHKERRQ(ierr);

  ierr = VecGetArray(locX, &locX_array); CHKERRQ(ierr);
  for (PetscInt n = 0; n < ctx->n_perio; n++) {
    PetscInt       n_master, n_slave;
    const PetscInt *master, *slave;
    PetscReal      *buffer_array;

    ierr = ISGetSize(ctx->perio[n].master, &n_master);  CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->perio[n].master, &master); CHKERRQ(ierr);
    for (PetscInt i = 0; i < n_master; i++) {
      PetscReal *val;
      ierr = DMPlexPointLocalFieldRead(dm, master[i], 0, locX_array, &val);             CHKERRQ(ierr);
  		ierr = VecSetValuesBlockedLocal(ctx->perio[n].buffer, 1, &i, val, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(ctx->perio[n].master, &master);  CHKERRQ(ierr);
    ierr = VecAssemblyBegin(ctx->perio[n].buffer);           CHKERRQ(ierr);
    ierr = VecAssemblyEnd(ctx->perio[n].buffer);             CHKERRQ(ierr);
    ierr = VecGetArray(ctx->perio[n].buffer, &buffer_array); CHKERRQ(ierr);

    ierr = ISGetSize(ctx->perio[n].slave, &n_slave);  CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->perio[n].slave, &slave); CHKERRQ(ierr);
    for (PetscInt i = 0; i < n_slave; i++) {
      PetscReal *val;
      ierr = DMPlexPointLocalFieldRef(dm, slave[i], 0, locX_array, &val); CHKERRQ(ierr);
      for (PetscInt k = 0; k < Nc; k++) val[k] = buffer_array[Nc * i + k];
    }
    ierr = VecRestoreArray(ctx->perio[n].buffer, &buffer_array); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(locX, &locX_array); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode MeshReconstructGradientsFVM(DM dm, Vec locX, Vec grad){
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecZeroEntries(grad); CHKERRQ(ierr);

  MeshCtx           ctx;
  Vec               cellgeom;
  DM                dmGrad, dmCell;
  const PetscScalar *x, *cellgeom_a;
  PetscScalar       *gr;
  PetscInt          cStart, dim, Nc;
  PetscLimiter      lim;
  PetscLimiterType  limType;
  { // Getting mesh data
    PetscFV      fvm;
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);     CHKERRQ(ierr);
    ierr = DMPlexGetDataFVM(dm, fvm, &cellgeom, NULL, NULL); CHKERRQ(ierr);
    ierr = VecGetDM(cellgeom, &dmCell);                      CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom, &cellgeom_a);           CHKERRQ(ierr);
    ierr = PetscFVGetLimiter(fvm, &lim);                     CHKERRQ(ierr);
    ierr = PetscLimiterGetType(lim, &limType);               CHKERRQ(ierr);
    if (!strcmp(limType, PETSCLIMITERZERO)) PetscFunctionReturn(0);
    ierr = DMGetDimension(dm, &dim);                         CHKERRQ(ierr);
    ierr = PetscFVGetNumComponents(fvm, &Nc);                CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);     CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, &ctx);                CHKERRQ(ierr);
    ierr = VecGetArrayRead(locX, &x);                        CHKERRQ(ierr);
    ierr = VecGetDM(grad, &dmGrad);                          CHKERRQ(ierr);
    ierr = VecGetArray(grad, &gr);                           CHKERRQ(ierr);
  }

  for (PetscInt n = 0; n < ctx->n_cell; n++) { // Computing cell gradient
    PetscInt       numNeighbors, c = cStart + n;
    const PetscInt *neighborhood;
    PetscScalar    *cx, *ncx, *cgrad;

    ierr = ISGetSize(ctx->CellCtx[n].neighborhood, &numNeighbors);    CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->CellCtx[n].neighborhood, &neighborhood); CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, c, x, &cx);                       CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dmGrad, c, gr, &cgrad);               CHKERRQ(ierr);
    for (PetscInt i = 0; i < numNeighbors; i++){
        ierr = DMPlexPointLocalRead(dm, neighborhood[i], x, &ncx); CHKERRQ(ierr);
        for (PetscInt j = 0; j < Nc; j++) {
          PetscScalar delta = ncx[j] - cx[j];
          for (PetscInt k = 0; k < dim; k++) cgrad[j * dim + k] += delta * ctx->CellCtx[n].grad_coeff[dim * i + k];
        }
    }
  }

  for (PetscInt n = (!strcmp(limType, PETSCLIMITERNONE)) ? ctx->n_cell : 0; n < ctx->n_cell; n++) { // Limit cell gradient
    const PetscInt  *faces, *fcells;
    PetscInt        nface, c = cStart + n, nc;
    PetscScalar     *cx, *ncx, *cgrad;
    PetscFVCellGeom *cg, *ncg;
    PetscReal cellPhi[Nc];
    ierr = DMPlexGetConeSize(dm, c, &nface);                 CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &faces);                     CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, c, x, &cx);              CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dmGrad, c, gr, &cgrad);      CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, c, cellgeom_a, &cg); CHKERRQ(ierr);
    for (PetscInt i = 0; i < Nc; i++) cellPhi[i] = PETSC_MAX_REAL;
    for (PetscInt f = 0; f < nface; f++) {
      PetscReal       dx[dim];

      ierr = DMPlexGetSupport(dm, faces[f], &fcells);            CHKERRQ(ierr);
      nc = (c == fcells[0]) ? fcells[1] : fcells[0];
      ierr = DMPlexPointLocalRead(dm, nc, x, &ncx);              CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, nc, cellgeom_a, &ncg); CHKERRQ(ierr);
      for (PetscInt j = 0; j < dim; j++) dx[j] = ncg->centroid[j] - cg->centroid[j];
      for (PetscInt i = 0; i < Nc; i++) {
        PetscReal denom = 0, phi;
        for (PetscInt j = 0; j < dim; j++) denom += cgrad[i * dim + j] * dx[j];
        ierr = PetscLimiterLimit(lim, (ncx[i] - cx[i]) / (2 * denom), &phi); CHKERRQ(ierr);
        cellPhi[i] = PetscMin(cellPhi[i], phi);
      }
    }
    for (PetscInt i = 0; i < Nc; i++){
      for (PetscInt j = 0; j < dim; j++) cgrad[i * dim + j] *= cellPhi[i];
    }
  }

  { // Cleanup
    ierr = VecRestoreArrayRead(cellgeom, &cellgeom_a); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locX, &x);              CHKERRQ(ierr);
    ierr = VecRestoreArray(grad, &gr);                 CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
