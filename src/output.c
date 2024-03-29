#include "output.h"

/*
  TODO
  Takes a single componant vector, casts it to the first component of a mesh based vector, and views it
*/
#include "spatial.h"
static PetscErrorCode DrawVecOnDM(Vec v, DM dm, PetscViewer viewer, const char *name){
  PetscErrorCode  ierr;
  Vec             v_dm;
  const PetscReal *v_data;
  PetscReal       *v_dm_data;
  PetscInt        n1, n2, Nc;
  PetscBool       flg;
  char            val[64];
  PetscFV         fvm;

  char       *compNameSave;
  const char *compName;

  PetscFunctionBeginUser;
  ierr = PetscOptionsGetString(NULL, NULL, "-draw_comp", val, sizeof(val), &flg); CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL, "-draw_comp", "0");                           CHKERRQ(ierr);

  ierr = MeshCreateGlobalVector(dm, &v_dm); CHKERRQ(ierr);
  ierr = VecGetLocalSize(v, &n1);           CHKERRQ(ierr);
  ierr = VecGetLocalSize(v_dm, &n2);        CHKERRQ(ierr);
  Nc = n2 / n1;
  ierr = VecGetArrayRead(v, &v_data);       CHKERRQ(ierr);
  ierr = VecGetArray(v_dm, &v_dm_data);     CHKERRQ(ierr);
  for (PetscInt i = 0; i < n1; i++) {v_dm_data[Nc * i] = v_data[i];}
  ierr = VecRestoreArray(v_dm, &v_dm_data); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v, &v_data);   CHKERRQ(ierr);

  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetComponentName(fvm, 0, &compName);   CHKERRQ(ierr);
  ierr = PetscStrallocpy(compName, &compNameSave);     CHKERRQ(ierr);
  ierr = PetscFVSetComponentName(fvm, 0, name);        CHKERRQ(ierr);

  ierr = VecView(v_dm, viewer); CHKERRQ(ierr);
  ierr = VecDestroy(&v_dm);     CHKERRQ(ierr);

  ierr = PetscFVSetComponentName(fvm, 0, compNameSave); CHKERRQ(ierr);
  ierr = PetscFree(compNameSave);                       CHKERRQ(ierr);

  if (flg) {
    ierr = PetscOptionsSetValue(NULL, "-draw_comp", val); CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsClearValue(NULL, "-draw_comp");    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscErrorCode MonitorAscii_MinMax(TS ts, PetscInt steps, PetscReal time, Vec u, void *ctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *mctx = (struct MonitorCtx*) ctx;

  PetscFunctionBeginUser;
  if (steps % mctx->n_iter != 0) PetscFunctionReturn(0);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%3d  time %8.4g  ", steps, time); CHKERRQ(ierr);

  DM       dm;
  PetscFV  fvm;
  PetscInt Nc;
  ierr = VecGetDM(u, &dm);                             CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);            CHKERRQ(ierr);
  for (PetscInt comp = 0; comp < Nc; comp++) {
    PetscReal  min, max;
    const char *compName;
    ierr = VecStrideMin(u, comp, NULL, &min);                                              CHKERRQ(ierr);
    ierr = VecStrideMax(u, comp, NULL, &max);                                              CHKERRQ(ierr);
    ierr = PetscFVGetComponentName(fvm, comp, &compName);                                  CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s : [% 10.4g, % 10.4g], ", compName, min, max); CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\033[2D\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorAscii_Res(TS ts, PetscInt steps, PetscReal time, Vec u, void *ctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *mctx = (struct MonitorCtx*) ctx;

  PetscFunctionBeginUser;
  if (steps % mctx->n_iter != 0) PetscFunctionReturn(0);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%4d  time %12.4g  ", steps, time); CHKERRQ(ierr);

  Vec flux;
  ierr = VecDuplicate(u, &flux);                  CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts, time, u, flux); CHKERRQ(ierr);

  DM       dm;
  PetscFV  fvm;
  PetscInt Nc, size;
  ierr = VecGetDM(u, &dm);                             CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);            CHKERRQ(ierr);
  ierr = VecGetLocalSize(flux, &size);                 CHKERRQ(ierr);
  for (PetscInt comp = 0; comp < Nc; comp++) {
    PetscReal  norm;
    const char *compName;
    ierr = VecStrideNorm(flux, comp, NORM_INFINITY, &norm);                 CHKERRQ(ierr);
    ierr = PetscFVGetComponentName(fvm, comp, &compName);                   CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s : % 10.4g, ", compName, norm); CHKERRQ(ierr);
  }
  ierr = VecDestroy(&flux);                          CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\033[2D\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorDraw(TS ts, PetscInt steps, PetscReal time, Vec u, void *ctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *mctx = (struct MonitorCtx*) ctx;

  PetscFunctionBeginUser;
  if (steps % mctx->n_iter != 0) PetscFunctionReturn(0);
  ierr = VecView(u, mctx->viewer); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "physics.h"
PetscErrorCode MonitorDrawNormU(TS ts, PetscInt steps, PetscReal time, Vec u, void *ctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *mctx = (struct MonitorCtx*) ctx;
  DM                dm;
  Vec               y;

  PetscFunctionBeginUser;
  if (steps % mctx->n_iter != 0) PetscFunctionReturn(0);

  ierr = VecGetDM(u, &dm);                                    CHKERRQ(ierr);
  ierr = VecApplyFunctionComponents(u, &y, mach, mctx->phys); CHKERRQ(ierr);
  ierr = DrawVecOnDM(y, dm, mctx->viewer, "|| u ||");         CHKERRQ(ierr);
  ierr = VecDestroy(&y);                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "view.h"
PetscErrorCode MonitorDrawGrad(TS ts, PetscInt steps, PetscReal time, Vec u, void *ctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *mctx = (struct MonitorCtx*) ctx;
  DM                dm, dmGrad;
  Vec               grad;
  PetscFV           fvm;

  PetscFunctionBeginUser;
  if (steps % mctx->n_iter != 0) PetscFunctionReturn(0);

  ierr = VecGetDM(u, &dm);                               CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);   CHKERRQ(ierr);
  ierr = DMPlexGetDataFVM(dm, fvm, NULL, NULL, &dmGrad); CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dmGrad, steps, time); CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dmGrad, &grad);                                 CHKERRQ(ierr);
  ierr = VecSetOperation(grad, VECOP_VIEW, (void (*)(void)) VecView_Mesh); CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(grad, "grad_");                               CHKERRQ(ierr);

  if (!steps) {
    Vec locX;
    ierr = DMGetLocalVector(dm, &locX);                                                           CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, u, INSERT_VALUES, locX);                                      CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, u, INSERT_VALUES, locX);                                        CHKERRQ(ierr);
    ierr = GlobalConservativeToLocalPrimitive_Endhook(dm, NULL, INSERT_VALUES, locX, mctx->phys); CHKERRQ(ierr);
    ierr = MeshReconstructGradientsFVM(dm, locX, grad);                                           CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locX);                                                       CHKERRQ(ierr);
  }

  ierr = VecView(grad, mctx->viewer); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmGrad, &grad); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorDEBUG(TS ts, PetscInt steps, PetscReal time, Vec u, void *ctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *mctx = (struct MonitorCtx*) ctx;

  PetscFunctionBeginUser;
  if (steps % mctx->n_iter != 0) PetscFunctionReturn(0);

  PetscReal dt;
  ierr = TSGetTimeStep(ts, &dt);                                                       CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%3d  time %8.4g  dt = %e\n", steps, time, dt); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode MonitorVTK(TS ts, PetscInt steps, PetscReal time, Vec u, void *ctx){
  PetscErrorCode    ierr;
  struct MonitorCtx *mctx = (struct MonitorCtx*) ctx;

  PetscFunctionBeginUser;
  if (steps % mctx->n_iter != 0) PetscFunctionReturn(0);
  ierr = TSMonitorSolutionVTK(ts, steps, time, u, (void*) mctx->output_filename_template); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
