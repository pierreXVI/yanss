#include "spatial.h"
#include "private_impl.h"


PetscErrorCode MeshDestroy(DM *mesh){
  PetscErrorCode ierr;
  PetscFV        fvm;

  PetscFunctionBeginUser;
  ierr = DMGetField(*mesh, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fvm);                                  CHKERRQ(ierr);
  ierr = DMDestroy(mesh);                                       CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MeshLoadFromFile(MPI_Comm comm, const char *filename, DM *mesh){
  PetscErrorCode ierr;
  DM             foo_dm;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, mesh);            CHKERRQ(ierr);
  ierr = DMViewFromOptions(*mesh, PETSC_NULL, "-dm_view_orig");             CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(*mesh, PETSC_TRUE, PETSC_FALSE);               CHKERRQ(ierr);
  ierr = DMPlexDistribute(*mesh, 1, PETSC_NULL, &foo_dm);                   CHKERRQ(ierr);
  if (foo_dm) {
    ierr = DMDestroy(mesh);                                                 CHKERRQ(ierr);
    *mesh = foo_dm;
  }
  ierr = DMSetFromOptions(*mesh);                                           CHKERRQ(ierr);
  ierr = DMPlexConstructGhostCells(*mesh, PETSC_NULL, PETSC_NULL, &foo_dm); CHKERRQ(ierr);
  ierr = DMDestroy(mesh);                                                   CHKERRQ(ierr);
  *mesh = foo_dm;
  ierr = PetscObjectSetName((PetscObject) *mesh, "Mesh");                   CHKERRQ(ierr);

  PetscFV fvm;
  ierr = PetscFVCreate(comm, &fvm);                         CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fvm, "FV Model"); CHKERRQ(ierr);
  ierr = DMAddField(*mesh, PETSC_NULL, (PetscObject) fvm);  CHKERRQ(ierr);

  char      opt[] = "____";
  PetscBool flag;
  PetscInt  numGhostCells;
  ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-dm_view", opt, sizeof(opt), PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscStrcmp(opt, "draw", &flag);                                                         CHKERRQ(ierr);
  if (flag) {ierr = DMPlexHideGhostCells(*mesh, &numGhostCells);                                  CHKERRQ(ierr);}
  ierr = DMViewFromOptions(*mesh, PETSC_NULL, "-dm_view");                                        CHKERRQ(ierr);
  if (flag) {ierr = DMPlexRestoreGhostCells(*mesh, numGhostCells);                                CHKERRQ(ierr);}

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



PetscErrorCode MeshComputeBoundary(DM dm, PetscReal time, Vec locX, Vec locX_t, void *user){
  PetscErrorCode ierr;
  IS *is_perio = (IS *) user;

  PetscFunctionBeginUser;

  PetscInt size;
  const PetscInt *master, *slave;
  PetscReal *values;
  ierr = ISGetSize(is_perio[0], &size); CHKERRQ(ierr);
  ierr = ISGetIndices(is_perio[0], &master); CHKERRQ(ierr);
  ierr = ISGetIndices(is_perio[1], &slave); CHKERRQ(ierr);
  ierr = VecGetArray(locX, &values); CHKERRQ(ierr);

  PetscFV  fvm;
  PetscInt Nc;
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);                  CHKERRQ(ierr);

  PetscReal *xI, *xG;
  for (PetscInt i = 0; i < size; i++) {
    ierr = DMPlexPointLocalRead(dm, master[i], values, &xI); CHKERRQ(ierr);
    ierr = DMPlexPointLocalFieldRef(dm, slave[i], 0, values, &xG); CHKERRQ(ierr);
    for (PetscInt j = 0; j < Nc; j++) xG[j] = xI[j];
  }

  ierr = VecRestoreArray(locX, &values); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_perio[1], &slave); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_perio[0], &master); CHKERRQ(ierr);

  ierr = DMPlexTSComputeBoundary(dm, time, locX, locX_t, user); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MeshSetupPeriodicBoundary(DM dm, IS foo[2]){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  IS is_master, is_slave;
  PetscInt n_master, n_slave;
  const PetscInt *idx_master, *idx_slave;

  ierr = DMGetStratumIS(dm, "Face Sets", 20, &is_master); CHKERRQ(ierr);
  ierr = DMGetStratumIS(dm, "Face Sets", 30, &is_slave); CHKERRQ(ierr);
  ISGetSize(is_master, &n_master);
  ISGetSize(is_master, &n_slave);
  if (n_master != n_slave) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Different number of faces on master (%d) ans slave (%d)\n", n_master, n_slave);
  ISGetIndices(is_master, &idx_master);
  ISGetIndices(is_slave, &idx_slave);

  PetscInt *master, *slave;
  PetscMalloc1(n_master, &master);
  PetscMalloc1(n_master, &slave);

  for (PetscInt i = 0; i < n_master; i++) {

  }


  ierr = ISRestoreIndices(is_master, &idx_master); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_slave, &idx_slave); CHKERRQ(ierr);

  // ISCreateGeneral(PETSC_COMM_WORLD, n_master, master, &foo[0])

  PetscFunctionReturn(0);
}
