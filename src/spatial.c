#include "spatial.h"
#include "private_impl.h"


PetscErrorCode MeshDestroy(Mesh *mesh){
  PetscErrorCode ierr;
  PetscFV        fvm;

  PetscFunctionBeginUser;
  ierr = DMGetField((*mesh)->dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fvm);                                        CHKERRQ(ierr);
  ierr = DMDestroy(&(*mesh)->dm);                                     CHKERRQ(ierr);
  ierr = PetscFree(mesh);                                             CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MeshLoadFromFile(MPI_Comm comm, const char *filename, Mesh *mesh){
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
  PetscScalar       val_y[size];
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



PetscErrorCode MeshDMComputeBoundary(DM dm, PetscReal time, Vec locX, Vec F, void *user){
  PetscErrorCode ierr;
  IS *masterSlave = (IS *) user;

  PetscFunctionBeginUser;

  PetscInt size;
  const PetscInt *master, *slave;
  PetscReal *values;
  ierr = ISGetSize(masterSlave[0], &size); CHKERRQ(ierr);
  ierr = ISGetIndices(masterSlave[0], &master); CHKERRQ(ierr);
  ierr = ISGetIndices(masterSlave[1], &slave); CHKERRQ(ierr);
  ierr = VecGetArray(locX, &values); CHKERRQ(ierr);



  PetscFV  fvm;
  PetscInt Nc;
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);                  CHKERRQ(ierr);

  PetscReal *xI, *xG;
  for (PetscInt i = 0; i < size; i++) {
    ierr = DMPlexPointLocalRead(dm, master[i], values, &xI); CHKERRQ(ierr);
    ierr = DMPlexPointLocalFieldRef(dm, slave[i], 0, values, &xG); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "%d -> %d (%f)\n", master[i], slave[i], xI[0]);
    for (PetscInt j = 0; j < Nc; j++) xG[j] = xI[j];
  }

  ierr = VecRestoreArray(locX, &values); CHKERRQ(ierr);
  ierr = ISRestoreIndices(masterSlave[1], &slave); CHKERRQ(ierr);
  ierr = ISRestoreIndices(masterSlave[0], &master); CHKERRQ(ierr);

  DMPlexTSComputeRHSFunctionFVM(dm, time, locX, F, user);

  PetscFunctionReturn(0);
}

PetscErrorCode MeshSetupPeriodicBoundary(Mesh mesh, PetscInt bc_from, PetscInt bc_to, PetscReal disp[]){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscInt dim;
  ierr = DMGetDimension(mesh->dm, &dim); CHKERRQ(ierr);

  IS             is_from, is_to;
  PetscInt       n_from, n_to;
  const PetscInt *idx_from, *idx_to;

  PetscPrintf(PETSC_COMM_WORLD, "Boundary %d -> %d\n", bc_from, bc_to);
  ierr = DMGetStratumIS(mesh->dm, "Face Sets", bc_from, &is_from); CHKERRQ(ierr);
  ierr = DMGetStratumIS(mesh->dm, "Face Sets", bc_to, &is_to);     CHKERRQ(ierr);
  ierr = ISGetSize(is_from, &n_from);                        CHKERRQ(ierr);
  ierr = ISGetSize(is_to, &n_to);                            CHKERRQ(ierr);
  if (n_from != n_to) SETERRQ4(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Different number of faces on boundaries %d (%d) and %d (%d)", bc_from, n_from, bc_to, n_to);
  ierr = ISGetIndices(is_from, &idx_from);                   CHKERRQ(ierr);
  ierr = ISGetIndices(is_to, &idx_to);                       CHKERRQ(ierr);


  PetscInt ghoscCStart, ghostCEnd;
  ierr = DMPlexGetGhostCellStratum(mesh->dm, &ghoscCStart, &ghostCEnd); CHKERRQ(ierr);

  PetscInt master[n_from], slave[n_to];

  for (PetscInt i = 0; i < n_from; i++) {
    PetscInt match_DEBUG=0;

    PetscReal c_from[dim], c_to[dim];
    ierr = DMPlexComputeCellGeometryFVM(mesh->dm, idx_from[i], PETSC_NULL, c_from, PETSC_NULL); CHKERRQ(ierr);
    for (PetscInt j = 0; j < n_to; j++) {
      ierr = DMPlexComputeCellGeometryFVM(mesh->dm, idx_to[j], PETSC_NULL, c_to, PETSC_NULL); CHKERRQ(ierr);

      PetscReal dist=0;
      for (PetscInt k = 0; k < dim; k++) dist += PetscSqr(c_to[k] - c_from[k] - disp[k]);

      if (dist < 1E-12) {
        match_DEBUG++;

        PetscInt nC_from, nC_to;
        DMPlexGetSupportSize(mesh->dm, idx_from[i], &nC_from);
        DMPlexGetSupportSize(mesh->dm, idx_to[j], &nC_to);
        if (nC_from != 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Support of size %d != 2", nC_from);
        if (nC_to != 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Support of size %d != 2", nC_to);

        PetscInt const *support_from, *support_to;
        DMPlexGetSupport(mesh->dm, idx_from[i], &support_from);
        DMPlexGetSupport(mesh->dm, idx_to[j], &support_to);
        master[i] = (ghoscCStart <= support_from[0] && support_from[0] < ghostCEnd) ? support_from[1] : support_from[0];
        slave[i] = (ghoscCStart <= support_to[0] && support_to[0] < ghostCEnd) ? support_to[0] : support_to[1];
      }
    }
    if (match_DEBUG != 1) SETERRQ4(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Cannot match face %d from boundary %d to single face on boundary %d, matched %d times", idx_from[i], bc_from, bc_to, match_DEBUG);
  }


  ierr = ISRestoreIndices(is_from, &idx_from); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_to, &idx_to); CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n_from, master, PETSC_COPY_VALUES, &mesh->masterSlave[0]); CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, n_from, slave, PETSC_COPY_VALUES, &mesh->masterSlave[1]); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}
