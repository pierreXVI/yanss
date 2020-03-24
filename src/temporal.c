#include "temporal.h"

static PetscErrorCode Monitor(TS ts, PetscInt stepnum, PetscReal time, Vec X, void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      xnorm;

  PetscFunctionBeginUser;
  ierr = VecNorm(X, NORM_INFINITY, &xnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "% 3D  time %8.4g  |x| %8.4g\n", stepnum, (double)time, (double)xnorm); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetTs(MPI_Comm comm, TS *ts, DM dm, PetscReal dt){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSCreate(comm, ts);                                   CHKERRQ(ierr);
  ierr = TSSetType(*ts, TSSSP);                                CHKERRQ(ierr);
  ierr = TSSetDM(*ts, dm);                                     CHKERRQ(ierr);
  ierr = TSSetMaxTime(*ts, 2.0);                               CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetTimeStep(*ts, dt);                               CHKERRQ(ierr);
  ierr = TSSetFromOptions(*ts);                                CHKERRQ(ierr);
  ierr = TSMonitorSet(*ts, Monitor, NULL, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
