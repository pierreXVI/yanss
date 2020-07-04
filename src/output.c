#include "output.h"

#include "private_impl.h"

#include "spatial.h"
PetscErrorCode DrawVecOnDM(Vec v, DM dm){
  PetscErrorCode     ierr;
  Vec v_dm;
  const PetscScalar *v_data;
  PetscScalar *v_dm_data;
  PetscInt n1, n2, Nc;

  PetscFunctionBeginUser;
  ierr = MeshCreateGlobalVector(dm, &v_dm); CHKERRQ(ierr);
  ierr = VecGetLocalSize(v, &n1);           CHKERRQ(ierr);
  ierr = VecGetLocalSize(v_dm, &n2);        CHKERRQ(ierr);
  Nc = n2 / n1;
  ierr = VecGetArrayRead(v, &v_data);       CHKERRQ(ierr);
  ierr = VecGetArray(v_dm, &v_dm_data);     CHKERRQ(ierr);
  for (PetscInt i = 0; i < n1; i++) {v_dm_data[Nc * i] = v_data[i];}
  ierr = VecRestoreArray(v_dm, &v_dm_data); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v, &v_data);   CHKERRQ(ierr);

  PetscBool flg;
  PetscInt displaycomp[Nc];
  char buffer[64];
  ierr = PetscOptionsGetIntArray(PETSC_NULL, PETSC_NULL, "-draw_comp", displaycomp, &Nc, &flg); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(PETSC_NULL, "-draw_comp", "0"); CHKERRQ(ierr);
  if (flg) {
    PetscPrintf(PETSC_COMM_WORLD, "Should rewrite option, %d\n", Nc);
    // ierr = PetscOptionsSetValue(PETSC_NULL, "-draw_comp", "1"); CHKERRQ(ierr);

  }

  PetscViewer viewer;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERDRAW); CHKERRQ(ierr);

  ierr = VecView(v_dm, viewer); CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  ierr = VecDestroy(&v_dm); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




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
    ierr = VecRestoreSubVector(u, is, &subv);                                              CHKERRQ(ierr);
    ierr = ISDestroy(&is);                                                                 CHKERRQ(ierr);
    ierr = PetscFVGetComponentName(fvm, comp, &compName);                                  CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s : [% 10.4g, % 10.4g], ", compName, min, max); CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\033[2D\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

PetscErrorCode IOMonitorDraw(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode     ierr;
  struct Monitor_ctx *ctx = (struct Monitor_ctx*) mctx;

  PetscFunctionBeginUser;
  if (steps % ctx->n_iter != 0) PetscFunctionReturn(0);
  ierr = VecView(u, ctx->viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IOMonitorDrawNormU(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode     ierr;
  struct Monitor_ctx *ctx = (struct Monitor_ctx*) mctx;

  PetscFunctionBeginUser;
  if (steps % ctx->n_iter != 0) PetscFunctionReturn(0);

  DM       dm;
  PetscFV  fvm;
  PetscInt Nc, size;
  IS       is_x, is_y;
  Vec      u_x, u_y;

  ierr = VecGetDM(u, &dm);                    CHKERRQ(ierr);

  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);                  CHKERRQ(ierr);
  ierr = VecGetLocalSize(u, &size);                          CHKERRQ(ierr);

  ierr = ISCreateStride(PetscObjectComm((PetscObject) u), size / Nc, 1, Nc, &is_x); CHKERRQ(ierr);
  // ierr = ISCreateStride(PetscObjectComm((PetscObject) u), size / Nc, 2, Nc, &is_y); CHKERRQ(ierr);
  ierr = VecGetSubVector(u, is_x, &u_x);                                            CHKERRQ(ierr);
  // ierr = VecGetSubVector(u, is_y, &u_y);                                            CHKERRQ(ierr);
  // ierr = VecPointwiseMult(u_x, u_x, u_x);                                           CHKERRQ(ierr);
  // ierr = VecPointwiseMult(u_y, u_y, u_y);                                           CHKERRQ(ierr);
  // ierr = VecAXPY(u_x, 1, u_y);                                                      CHKERRQ(ierr);
  // ierr = VecSqrtAbs(u_x);                                                           CHKERRQ(ierr);

  ierr = DrawVecOnDM(u_x, dm); CHKERRQ(ierr);


  // ierr = VecView(ud, PETSC_VIEWER_DRAW_WORLD);      CHKERRQ(ierr);

  ierr = VecRestoreSubVector(u, is_x, &u_x);                                        CHKERRQ(ierr);
  // ierr = VecRestoreSubVector(u, is_y, &u_y);                                        CHKERRQ(ierr);
  ierr = ISDestroy(&is_x);                                                          CHKERRQ(ierr);
  // ierr = ISDestroy(&is_y);                                                          CHKERRQ(ierr);



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
