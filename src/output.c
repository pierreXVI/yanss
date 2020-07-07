#include "output.h"

#include "spatial.h"
#include "private_impl.h"
PetscErrorCode DrawVecOnDM(Vec v, DM dm, PetscViewer viewer){
  PetscErrorCode    ierr;
  Vec               v_dm;
  const PetscScalar *v_data;
  PetscScalar       *v_dm_data;
  PetscInt          n1, n2, Nc;
  PetscBool         flg;
  char              val[64];
  const char        *name;

  PetscFunctionBeginUser;
  ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-draw_comp", val, sizeof(val), &flg); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(PETSC_NULL, "-draw_comp", "0");                                 CHKERRQ(ierr);

  ierr = MeshCreateGlobalVector(dm, &v_dm); CHKERRQ(ierr);
  ierr = VecGetLocalSize(v, &n1);           CHKERRQ(ierr);
  ierr = VecGetLocalSize(v_dm, &n2);        CHKERRQ(ierr);
  Nc = n2 / n1;
  ierr = VecGetArrayRead(v, &v_data);       CHKERRQ(ierr);
  ierr = VecGetArray(v_dm, &v_dm_data);     CHKERRQ(ierr);
  for (PetscInt i = 0; i < n1; i++) {v_dm_data[Nc * i] = v_data[i];}
  ierr = VecRestoreArray(v_dm, &v_dm_data); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v, &v_data);   CHKERRQ(ierr);

  ierr = PetscObjectGetName((PetscObject) v, &name);   CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) v_dm, name); CHKERRQ(ierr);

  ierr = VecView(v_dm, viewer);             CHKERRQ(ierr);
  ierr = VecDestroy(&v_dm);                 CHKERRQ(ierr);

  if (flg) {
    ierr = PetscOptionsSetValue(PETSC_NULL, "-draw_comp", val); CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsClearValue(PETSC_NULL, "-draw_comp");    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscErrorCode IOMonitorAscii_MinMax(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *ctx = (struct MonitorCtx*) mctx;

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
    ierr = VecRestoreSubVector(u, is, &subv);                                              CHKERRQ(ierr);
    ierr = ISDestroy(&is);                                                                 CHKERRQ(ierr);
    ierr = PetscFVGetComponentName(fvm, comp, &compName);                                  CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s : [% 10.4g, % 10.4g], ", compName, min, max); CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\033[2D\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IOMonitorAscii_Res(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *ctx = (struct MonitorCtx*) mctx;

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
  ierr = VecDestroy(&flux);                          CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\033[2D\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IOMonitorDraw(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *ctx = (struct MonitorCtx*) mctx;

  PetscFunctionBeginUser;
  if (steps % ctx->n_iter != 0) PetscFunctionReturn(0);
  ierr = VecView(u, ctx->viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode normU(PetscInt Nc, const PetscScalar *x, PetscScalar *y, void *ctx){
  PetscInt *dim = (PetscInt*) ctx;

  PetscFunctionBeginUser;
  *y = 0;
  for (PetscInt i = 0; i < *dim; i++) *y += PetscSqr(x[1 + i]);
  *y = PetscSqrtReal(*y);
  PetscFunctionReturn(0);
}

PetscErrorCode IOMonitorDrawNormU(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *ctx = (struct MonitorCtx*) mctx;
  PetscInt          dim;
  DM                dm;
  Vec               y;

  PetscFunctionBeginUser;
  if (steps % ctx->n_iter != 0) PetscFunctionReturn(0);

  ierr = VecGetDM(u, &dm);                           CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);                   CHKERRQ(ierr);
  ierr = VecApplyFunctionFields(u, &y, normU, &dim); CHKERRQ(ierr);
  ierr = DrawVecOnDM(y, dm, ctx->viewer);            CHKERRQ(ierr);
  ierr = VecDestroy(&y);                             CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode IOMonitorDEBUG(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *ctx = (struct MonitorCtx*) mctx;

  PetscFunctionBeginUser;
  if (steps % ctx->n_iter != 0) PetscFunctionReturn(0);

  PetscReal dt;
  ierr = TSGetTimeStep(ts, &dt);                                                       CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%3d  time %8.4g  dt = %e\n", steps, time, dt); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
