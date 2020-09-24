#include "temporal.h"
#include "input.h"
#include "output.h"


struct MonitorFunctionList {
  const char      *name;
  PetscErrorCode  (*func)(TS, PetscInt, PetscReal, Vec, void*);
  PetscViewerType type;
} MonitorList[] = {{"Debug",        IOMonitorDEBUG,        NULL},
                   {"Ascii_Res",    IOMonitorAscii_Res,    NULL},
                   {"Ascii_MinMax", IOMonitorAscii_MinMax, NULL},
                   {"Draw",         IOMonitorDraw,         PETSCVIEWERDRAW},
                   {"Draw_NormU",   IOMonitorDrawNormU,    PETSCVIEWERDRAW},
                   {"Draw_Grad",    IOMonitorDrawGrad,     PETSCVIEWERDRAW},
                   {NULL,           NULL,                  NULL}};

/*
  Wrapper to free a Monitor Context
*/
static PetscErrorCode MonitorCtxDestroy(void **ctx) {
  PetscErrorCode    ierr;
  struct MonitorCtx *mctx = (struct MonitorCtx*) *ctx;

  PetscFunctionBeginUser;
  ierr = PetscViewerDestroy(&mctx->viewer); CHKERRQ(ierr);
  ierr = PetscFree(mctx);                   CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode TsCreate_User(MPI_Comm comm, TS *ts, const char *filename, DM dm, Physics phys, PetscReal cfl){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSCreate(comm, ts);                                   CHKERRQ(ierr);
  ierr = TSSetDM(*ts, dm);                                     CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);

  for (PetscInt i = 0; MonitorList[i].name; i++) {
    PetscBool set;
    PetscInt  n_iter;
    ierr = IOLoadMonitorOptions(filename, MonitorList[i].name, &set, &n_iter); CHKERRQ(ierr);
    if (set && n_iter > 0) {
      struct MonitorCtx *ctx;
      ierr = PetscNew(&ctx); CHKERRQ(ierr);
      ctx->n_iter = n_iter;
      ctx->phys = phys;
      if (MonitorList[i].type) {
        ierr = PetscViewerCreate(comm, &ctx->viewer);                          CHKERRQ(ierr);
        ierr = PetscViewerSetType(ctx->viewer, MonitorList[i].type);           CHKERRQ(ierr);
      }
      ierr = TSMonitorSet(*ts, MonitorList[i].func, ctx, MonitorCtxDestroy);   CHKERRQ(ierr);
    }
  }

  PetscReal dt, minRadius, norm2 = 0;
  ierr = DMPlexTSGetGeometryFVM(dm, NULL, NULL, &minRadius); CHKERRQ(ierr);
  for (PetscInt i = 0; i < phys->dim; i++) norm2 += PetscSqr(phys->init[1 + i]);
  dt = cfl * minRadius / (PetscSqrtReal(phys->gamma * phys->init[phys->dim + 1] / phys->init[0]) + PetscSqrtReal(norm2));
  ierr = TSSetTimeStep(*ts, dt); CHKERRQ(ierr);

  ierr = TSSetFromOptions(*ts);                    CHKERRQ(ierr);
  ierr = TSViewFromOptions(*ts, NULL, "-ts_view"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
