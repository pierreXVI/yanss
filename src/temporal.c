#include "temporal.h"
#include "input.h"
#include "output.h"

struct MonitorFunctionList {
  const char     *name;
  PetscErrorCode (*func)(TS, PetscInt, PetscReal, Vec, void*);
} MonitorList[] = {{"Debug",        IOMonitorDEBUG},
                   {"Ascii_Res",    IOMonitorAscii_Res},
                   {"Ascii_MinMax", IOMonitorAscii_MinMax},
                   {"Draw",         IOMonitorDraw},
                   {"Draw_NormU",   IOMonitorDrawNormU},
                   {PETSC_NULL,     PETSC_NULL}};

static PetscErrorCode PetscFreeWrapper(void **mctx) {return PetscFree(*mctx);}


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
      ierr = TSMonitorSet(*ts, MonitorList[i].func, ctx, PetscFreeWrapper); CHKERRQ(ierr);
    }
  }

  PetscReal dt, minRadius, norm = 0;
  ierr = DMPlexTSGetGeometryFVM(dm, PETSC_NULL, PETSC_NULL, &minRadius); CHKERRQ(ierr);
  for (PetscInt i = 0; i < phys->dim; i++) {
    norm += PetscSqr(phys->init[1 + i]);
  }
  dt = cfl * minRadius / (PetscSqrtReal(phys->gamma * phys->init[phys->dim + 1] / phys->init[0]) + PetscSqrtReal(norm));
  PetscPrintf(PETSC_COMM_WORLD, "Dt = %g\n", dt);
  ierr = TSSetTimeStep(*ts, dt); CHKERRQ(ierr);

  ierr = TSSetFromOptions(*ts); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
