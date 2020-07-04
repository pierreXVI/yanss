#include "temporal.h"
#include "input.h"
#include "output.h"

struct MonitorFunctionList {
  const char      *name;
  PetscErrorCode  (*func)(TS, PetscInt, PetscReal, Vec, void*);
  PetscViewerType type;
} MonitorList[] = {{"Debug",        IOMonitorDEBUG,        PETSC_NULL},
                   {"Ascii_Res",    IOMonitorAscii_Res,    PETSC_NULL},
                   {"Ascii_MinMax", IOMonitorAscii_MinMax, PETSC_NULL },
                   {"Draw",         IOMonitorDraw,         PETSCVIEWERDRAW },
                   {"Draw_NormU",   IOMonitorDrawNormU,    PETSCVIEWERDRAW },
                   {PETSC_NULL,     PETSC_NULL,            PETSC_NULL}};


static PetscErrorCode PetscFreeWrapper(void **mctx) {
  struct Monitor_ctx *ctx = (struct Monitor_ctx*) *mctx;
  PetscViewerDestroy(&ctx->viewer);
  return PetscFree(*mctx);
}


PetscErrorCode MyTsCreate(MPI_Comm comm, TS *ts, const char *filename, DM dm, Physics phys, PetscReal cfl){
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
      struct Monitor_ctx *ctx;
      ierr = PetscNew(&ctx); CHKERRQ(ierr);
      ctx->n_iter = n_iter;
      if (MonitorList[i].type) {
        ierr = PetscViewerCreate(comm, &ctx->viewer); CHKERRQ(ierr);
        ierr = PetscViewerSetType(ctx->viewer, MonitorList[i].type); CHKERRQ(ierr);
      }
      ierr = TSMonitorSet(*ts, MonitorList[i].func, ctx, PetscFreeWrapper); CHKERRQ(ierr);
    }
  }

  PetscReal dt, minRadius, norm = 0;
  ierr = DMPlexTSGetGeometryFVM(dm, PETSC_NULL, PETSC_NULL, &minRadius); CHKERRQ(ierr);
  for (PetscInt i = 0; i < phys->dim; i++) {
    norm += PetscSqr(phys->init[1 + i]);
  }
  dt = cfl * minRadius / (PetscSqrtReal(phys->gamma * phys->init[phys->dim + 1] / phys->init[0]) + PetscSqrtReal(norm));
  ierr = TSSetTimeStep(*ts, dt); CHKERRQ(ierr);

  ierr = TSSetFromOptions(*ts); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
