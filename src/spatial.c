#include "spatial.h"
#include "input.h"
#include "private_impl.h"


/*
  Fills the IS mesh->perio to represent the mesh periodicity
  disp is the displacement from bc_1 to bc_2
  Both boundaries are removed from the label "Face Sets" of the mesh->dm
*/
static PetscErrorCode MeshSetPeriodicity(Mesh mesh, PetscInt bc_1, PetscInt bc_2, PetscReal *disp){
  PetscErrorCode ierr;
  IS             is_1, is_2;
  const PetscInt *faces_1, *faces_2;
  PetscInt       dim, ghoscCStart, ghostCEnd, nface, *cells_1, *cells_2;

  PetscFunctionBeginUser;

  ierr = DMGetDimension(mesh->dm, &dim);                                CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(mesh->dm, &ghoscCStart, &ghostCEnd); CHKERRQ(ierr);
  ierr = DMGetStratumIS(mesh->dm, "Face Sets", bc_1, &is_1);            CHKERRQ(ierr);
  ierr = DMGetStratumIS(mesh->dm, "Face Sets", bc_2, &is_2);            CHKERRQ(ierr);
  ierr = ISGetSize(is_1, &nface);                                       CHKERRQ(ierr);
  ierr = ISGetIndices(is_1, &faces_1);                                  CHKERRQ(ierr);
  ierr = ISGetIndices(is_2, &faces_2);                                  CHKERRQ(ierr);
  ierr = PetscMalloc1(2 * nface, &cells_1);                             CHKERRQ(ierr);
  ierr = PetscMalloc1(2 * nface, &cells_2);                             CHKERRQ(ierr);

  for (PetscInt i = 0; i < nface; i++) {
    PetscReal c_1[dim], c_2[dim], len;
    PetscBool found = PETSC_FALSE;

    ierr = DMPlexComputeCellGeometryFVM(mesh->dm, faces_1[i], &len, c_1, PETSC_NULL);         CHKERRQ(ierr);
    for (PetscInt j = 0; j < nface; j++) {
      ierr = DMPlexComputeCellGeometryFVM(mesh->dm, faces_2[j], PETSC_NULL, c_2, PETSC_NULL); CHKERRQ(ierr);
      PetscReal dist = 0;
      for (PetscInt k = 0; k < dim; k++) dist += PetscSqr(c_2[k] - c_1[k] - disp[k]);

      if (PetscSqrtReal(dist) / len < 1E-1) {
        found = PETSC_TRUE;

        PetscInt const *support_1, *support_2;
        ierr = DMPlexGetSupport(mesh->dm, faces_1[i], &support_1); CHKERRQ(ierr);
        ierr = DMPlexGetSupport(mesh->dm, faces_2[j], &support_2); CHKERRQ(ierr);

        if (ghoscCStart <= support_1[0] && support_1[0] < ghostCEnd) {
          cells_1[i] = support_1[1];
          cells_2[i + nface] = support_1[0];
        } else {
          cells_1[i] = support_1[0];
          cells_2[i + nface] = support_1[1];
        }

        if (ghoscCStart <= support_2[0] && support_2[0] < ghostCEnd) {
          cells_1[i + nface] = support_2[1];
          cells_2[i] = support_2[0];
        } else {
          cells_1[i + nface] = support_2[0];
          cells_2[i] = support_2[1];
        }
        break;
      }
    }

    if (!found) SETERRQ3(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Cannot find periodic face on boundary %d for face %d on boundary %d with given displacement", bc_2, faces_1[i], bc_1);
  }
  ierr = ISRestoreIndices(is_1, &faces_1); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is_2, &faces_2); CHKERRQ(ierr);
  ierr = ISDestroy(&is_1);                 CHKERRQ(ierr);
  ierr = ISDestroy(&is_2);                 CHKERRQ(ierr);


  IS is_list[2];

  is_list[0] = mesh->perio[0];
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) mesh->perio[0]), nface, cells_1, PETSC_OWN_POINTER, &is_list[1]); CHKERRQ(ierr);
  ierr = ISConcatenate(PetscObjectComm((PetscObject) mesh->perio[0]), 2, is_list, &(mesh->perio[0]));                    CHKERRQ(ierr);
  ierr = ISDestroy(&is_list[0]);                                                                                         CHKERRQ(ierr);
  ierr = ISDestroy(&is_list[1]);                                                                                         CHKERRQ(ierr);

  is_list[0] = mesh->perio[1];
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) mesh->perio[1]), nface, cells_2, PETSC_OWN_POINTER, &is_list[1]); CHKERRQ(ierr);
  ierr = ISConcatenate(PetscObjectComm((PetscObject) mesh->perio[1]), 2, is_list, &(mesh->perio[1]));                    CHKERRQ(ierr);
  ierr = ISDestroy(&is_list[0]);                                                                                         CHKERRQ(ierr);
  ierr = ISDestroy(&is_list[1]);                                                                                         CHKERRQ(ierr);

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
    for (PetscInt j = 0; j < Nc; j++) xG[j] = xI[j];
  }

  ierr = VecRestoreArray(locX, &values); CHKERRQ(ierr);
  ierr = ISRestoreIndices(masterSlave[1], &slave); CHKERRQ(ierr);
  ierr = ISRestoreIndices(masterSlave[0], &master); CHKERRQ(ierr);

  DMPlexTSComputeRHSFunctionFVM(dm, time, locX, F, user);

  PetscFunctionReturn(0);
}
