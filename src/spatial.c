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

  ierr = DMPlexDistribute(*dm, 1, NULL, &foo_dm);              CHKERRQ(ierr);
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
  ierr = PetscFVCreate(comm, &fvm);                CHKERRQ(ierr);
  ierr = DMAddField(*dm, NULL, (PetscObject) fvm); CHKERRQ(ierr);

  MeshCtx ctx;
  ierr = PetscNew(&ctx);                                                CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, ctx);                             CHKERRQ(ierr);
  ierr = DMTSSetRHSFunctionLocal(*dm, MeshComputeRHSFunctionFVM, NULL); CHKERRQ(ierr);

  ierr = MeshSetViewer(*dm);                         CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-mesh_view"); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode MeshSetUp(DM dm, Physics phys, const char *filename){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscFV fvm;
  { // Setting PetscFV
    PetscLimiter limiter;
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);      CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fvm, "FV Model"); CHKERRQ(ierr);
    ierr = PetscFVSetType(fvm, PETSCFVLEASTSQUARES);          CHKERRQ(ierr);
    ierr = PetscFVGetLimiter(fvm, &limiter);                  CHKERRQ(ierr);
    ierr = PetscLimiterSetType(limiter, PETSCLIMITERNONE);    CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvm, phys->dim);        CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(fvm);                        CHKERRQ(ierr);

    PetscFV  fvmGrad;
    PetscInt Nc;
    char buffer[64];
    ierr = PetscFVCreate(PetscObjectComm((PetscObject) dm), &fvmGrad); CHKERRQ(ierr);
    ierr = PetscFVGetNumComponents(fvm, &Nc);                          CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvmGrad, phys->dim);             CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvmGrad, phys->dim * Nc);           CHKERRQ(ierr);
    for (PetscInt i = 0; i < Nc; i++){
        const char *cname;
        ierr = PetscFVGetComponentName(fvm, i, &cname); CHKERRQ(ierr);
        for (PetscInt k = 0; k < phys->dim; k++) {
          ierr = PetscSNPrintf(buffer, sizeof(buffer),"d_%d %s", k, cname);   CHKERRQ(ierr);
          ierr = PetscFVSetComponentName(fvmGrad, phys->dim * i + k, buffer); CHKERRQ(ierr);
        }
      }

    DM dmGrad;
    ierr = DMPlexGetDataFVM(dm, fvm, NULL, NULL, &dmGrad);  CHKERRQ(ierr);
    ierr = DMAddField(dmGrad, NULL, (PetscObject) fvmGrad); CHKERRQ(ierr);
  }

  ierr = MeshSetPeriodicity(dm, filename); CHKERRQ(ierr);

  {
    PetscDS prob;
    DMLabel label;
    IS      is;
    const PetscInt *indices;
    ierr = DMCreateDS(dm);                                         CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);                                     CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, 0, phys->riemann_solver); CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 0, phys);                       CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "Face Sets", &label);                    CHKERRQ(ierr);
    ierr = DMLabelGetNumValues(label, &phys->nbc);                 CHKERRQ(ierr);
    ierr = PetscMalloc1(phys->nbc, &phys->bc_ctx);                 CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &is);                          CHKERRQ(ierr);
    ierr = ISGetIndices(is, &indices);                             CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);                                     CHKERRQ(ierr);

    PetscFunctionList bcList;
    ierr = BCRegister(&bcList); CHKERRQ(ierr);

    for (PetscInt i = 0; i < phys->nbc; i++) {
      void (*bcFunc)(void);

      phys->bc_ctx[i].phys = phys;
      ierr = IOLoadBC(filename, indices[i], phys->dim, phys->bc_ctx + i);       CHKERRQ(ierr);
      ierr = PetscFunctionListFind(bcList, phys->bc_ctx[i].type, &bcFunc);      CHKERRQ(ierr);

      if (!strcmp(phys->bc_ctx[i].type, "BC_DIRICHLET")) {
        PrimitiveToConservative(phys, phys->bc_ctx[i].val, phys->bc_ctx[i].val);
      }

      if (!bcFunc) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unknown boundary condition (%s)", phys->bc_ctx[i].type);
      ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, phys->bc_ctx[i].name, "Face Sets", 0, 0,
                                NULL, bcFunc, 1, indices + i, phys->bc_ctx + i); CHKERRQ(ierr);
    }
    ierr = PetscFunctionListDestroy(&bcList); CHKERRQ(ierr);
    ierr = ISRestoreIndices(is, &indices); CHKERRQ(ierr);
    ierr = ISDestroy(&is);                 CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);    CHKERRQ(ierr);

  }

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
static PetscErrorCode MeshSetPeriodicityCtx(DM dm, PetscInt bc_1, PetscInt bc_2, PetscReal *disp, struct PerioCtx *ctx){
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

    PetscInt nface_save;
    nface_save = nface1_loc;
    for (PetscInt i = 0; i < nface_save; i++) {
      PetscInt support_size;
      ierr = DMPlexGetSupportSize(dm, face1[i], &support_size); CHKERRQ(ierr);
      if (support_size != 2) nface1_loc--;
    }
    nface_save = nface2_loc;
    for (PetscInt j = 0; j < nface_save; j++) {
      PetscInt support_size;
      ierr = DMPlexGetSupportSize(dm, face2[j], &support_size); CHKERRQ(ierr);
      if (support_size != 2) nface2_loc--;
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
      PetscInt  loc, support_size;
      PetscReal val[dim];
      ierr = DMPlexGetSupportSize(dm, face1[i], &support_size);           CHKERRQ(ierr);
      if (support_size != 2) {i--; continue;}
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
      PetscInt  loc, support_size;
      PetscReal val[dim];
      ierr = DMPlexGetSupportSize(dm, face2[j], &support_size);           CHKERRQ(ierr);
      if (support_size != 2) {j--; continue;}
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

PetscErrorCode MeshSetPeriodicity(DM dm, const char *opt_filename) {
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
        ierr = MeshSetPeriodicityCtx(dm, master, bnd[i], disp, &ctxs[ctx->n_perio++]); CHKERRQ(ierr);
        ierr = PetscFree(disp);                                                        CHKERRQ(ierr);
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


PetscErrorCode MeshComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *ctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MeshInsertPeriodicValues(dm, locX); CHKERRQ(ierr);

  PetscFV fvm;
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);          CHKERRQ(ierr);
  ierr = DMPlexTSComputeRHSFunctionFVM(dm, time, locX, F, ctx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
