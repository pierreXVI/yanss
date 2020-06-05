#include "output.h"

/*
  Print for each field "field_name : min_value, max_value"
*/
PetscErrorCode IOMonitorAscii_MinMax(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode     ierr;
  struct Monitor_ctx *ctx = (struct Monitor_ctx*) mctx;

  PetscFunctionBeginUser;
  if (steps % ctx->n_iter != 0) PetscFunctionReturn(0);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%3d  time %8.4g  ", steps, time); CHKERRQ(ierr);

  DM       dm;
  PetscFV  fvm;
  PetscInt Nc, size;
  ierr = VecGetDM(u, &dm);                                   CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);                  CHKERRQ(ierr);
  ierr = VecGetLocalSize(u, &size);                          CHKERRQ(ierr);
  for (PetscInt comp = 0; comp < Nc; comp++) {
    IS         is;
    Vec        subv;
    PetscReal   min, max;
    const char *compName;
    ierr = ISCreateStride(PetscObjectComm((PetscObject) u), size / Nc, comp, Nc, &is);     CHKERRQ(ierr);
    ierr = VecGetSubVector(u, is, &subv);                                                  CHKERRQ(ierr);
    ierr = VecMin(subv, PETSC_NULL, &min);                                                 CHKERRQ(ierr);
    ierr = VecMax(subv, PETSC_NULL, &max);                                                 CHKERRQ(ierr);
    ierr = VecDestroy(&subv);                                                              CHKERRQ(ierr);
    ierr = ISDestroy(&is);                                                                 CHKERRQ(ierr);
    ierr = PetscFVGetComponentName(fvm, comp, &compName);                                  CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s : [% 10.4g, % 10.4g], ", compName, min, max); CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\033[2D\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Print for each field "field_name : min_flux_value, max_fluxvalue"
*/
PetscErrorCode IOMonitorAscii_Res(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode     ierr;
  struct Monitor_ctx *ctx = (struct Monitor_ctx*) mctx;

  PetscFunctionBeginUser;
  if (steps % ctx->n_iter != 0) PetscFunctionReturn(0);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%3d  time %8.4g  ", steps, time); CHKERRQ(ierr);

  Vec flux;
  ierr = VecDuplicate(u, &flux);                  CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts, time, u, flux); CHKERRQ(ierr);

  DM       dm;
  PetscFV  fvm;
  PetscInt Nc, size;
  ierr = VecGetDM(u, &dm);                                   CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);                  CHKERRQ(ierr);
  ierr = VecGetLocalSize(flux, &size);                       CHKERRQ(ierr);
  for (PetscInt comp = 0; comp < Nc; comp++) {
    IS         is;
    Vec        subv;
    PetscReal  norm;
    const char *compName;
    ierr = ISCreateStride(PetscObjectComm((PetscObject) flux), size / Nc, comp, Nc, &is);  CHKERRQ(ierr);
    ierr = VecGetSubVector(flux, is, &subv);                                               CHKERRQ(ierr);
    ierr = VecNorm(subv, NORM_INFINITY, &norm);                                            CHKERRQ(ierr);
    ierr = VecDestroy(&subv);                                                              CHKERRQ(ierr);
    ierr = ISDestroy(&is);                                                                 CHKERRQ(ierr);
    ierr = PetscFVGetComponentName(fvm, comp, &compName);                                  CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s : % 10.4g, ", compName, norm);                CHKERRQ(ierr);
  }
  ierr = VecDestroy(&flux);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\033[2D\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Draw the fields
*/
PetscErrorCode IOMonitorDraw(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode     ierr;
  struct Monitor_ctx *ctx = (struct Monitor_ctx*) mctx;

  PetscFunctionBeginUser;
  if (steps % ctx->n_iter != 0) PetscFunctionReturn(0);

  DM       dm;
  PetscInt numGhostCells;
  ierr = VecGetDM(u, &dm);                           CHKERRQ(ierr);
  ierr = DMPlexHideGhostCells(dm, &numGhostCells);   CHKERRQ(ierr);
  ierr = VecView(u, PETSC_VIEWER_DRAW_WORLD);        CHKERRQ(ierr);
  ierr = DMPlexRestoreGhostCells(dm, numGhostCells); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode IOMonitorDEBUG(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode     ierr;
  struct Monitor_ctx *ctx = (struct Monitor_ctx*) mctx;

  PetscFunctionBeginUser;
  if (steps % ctx->n_iter != 0) PetscFunctionReturn(0);

  PetscReal dt;
  ierr = TSGetTimeStep(ts, &dt); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%3d  time %8.4g  dt = %e\n", steps, time, dt); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
