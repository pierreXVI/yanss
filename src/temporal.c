#include "temporal.h"
#include "input.h"
#include "output.h"


struct MonitorFunctionList {
  const char      *name;
  PetscErrorCode  (*func)(TS, PetscInt, PetscReal, Vec, void*);
  PetscViewerType type;
} MonitorList[] = {{"Debug",        MonitorDEBUG,        NULL},
                   {"Ascii_Res",    MonitorAscii_Res,    NULL},
                   {"Ascii_MinMax", MonitorAscii_MinMax, NULL},
                   {"Draw",         MonitorDraw,         PETSCVIEWERDRAW},
                   {"Draw_NormU",   MonitorDrawNormU,    PETSCVIEWERDRAW},
                   {"Draw_Grad",    MonitorDrawGrad,     PETSCVIEWERDRAW},
                   {NULL,           NULL,                NULL}};

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


PetscErrorCode TSCreate_User(MPI_Comm comm, TS *ts, const char *filename, DM dm, Physics phys){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSCreate(comm, ts); CHKERRQ(ierr);
  ierr = TSSetDM(*ts, dm);   CHKERRQ(ierr);

  for (PetscInt i = 0; MonitorList[i].name; i++) {
    PetscBool set;
    PetscInt  n_iter;
    ierr = YAMLLoadMonitorOptions(filename, MonitorList[i].name, &set, &n_iter); CHKERRQ(ierr);
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

  const char *buffer, *loc = "Temporal";
  ierr = YAMLLoadVarFromLoc(filename, "cfl", 1, &loc, &buffer); CHKERRQ(ierr);
  PetscReal cfl = atof(buffer);
  ierr = PetscFree(buffer); CHKERRQ(ierr);

  PetscReal dt, minRadius, norm2 = 0;
  ierr = DMPlexGetGeometryFVM(dm, NULL, NULL, &minRadius); CHKERRQ(ierr);
  for (PetscInt i = 0; i < phys->dim; i++) norm2 += PetscSqr(phys->init[1 + i]);
  dt = cfl * minRadius / (PetscSqrtReal(phys->gamma * phys->init[phys->dim + 1] / phys->init[0]) + PetscSqrtReal(norm2));
  ierr = TSSetTimeStep(*ts, dt); CHKERRQ(ierr);

  ierr = TSSetFromOptions(*ts); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
