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

  ierr = DMTSSetBoundaryLocal(*mesh, DMPlexTSComputeBoundary, PETSC_NULL);          CHKERRQ(ierr);
  ierr = DMTSSetRHSFunctionLocal(*mesh, DMPlexTSComputeRHSFunctionFVM, PETSC_NULL); CHKERRQ(ierr);

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


PetscErrorCode VecGetFieldVectors(Vec x, PetscInt *Nc_p, Vec **fields){
  PetscErrorCode ierr;
  PetscInt       Nc, start, end;
  PetscFV        fvm;
  DM             dm;

  PetscFunctionBeginUser;
  ierr = VecGetDM(x, &dm);                                   CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);                  CHKERRQ(ierr);
  ierr = PetscMalloc1(Nc, fields);                           CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x, &start, &end);              CHKERRQ(ierr);

  IS  is;
  Vec v;
  for (PetscInt i = 0; i < Nc; i++) {
    ierr = ISCreateStride(PetscObjectComm((PetscObject) x), (end - start) / Nc, start + i, Nc, &is); CHKERRQ(ierr);
    ierr = VecGetSubVector(x, is, &v);                                                               CHKERRQ(ierr);
    ierr = VecDuplicate(v, &(*fields)[i]);                                                           CHKERRQ(ierr);
    ierr = VecCopy(v, (*fields)[i]);                                                                 CHKERRQ(ierr);
    ierr = VecRestoreSubVector(x, is, &v);                                                           CHKERRQ(ierr);
    ierr = ISDestroy(&is);                                                                           CHKERRQ(ierr);
  }
  if (Nc_p) *Nc_p = Nc;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroyFieldVectors(Vec x, Vec **fields){
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
