#include "temporal.h"
#include "spatial.h"

static PetscErrorCode Monitor(TS ts, PetscInt stepnum, PetscReal time, Vec x, void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      xnorm;
  PetscInt       numGhostCells;
  DM             mesh;

  PetscFunctionBeginUser;
  ierr = VecGetDM(x, &mesh); CHKERRQ(ierr);
  ierr = HideGhostCells(mesh, &numGhostCells); CHKERRQ(ierr);
  ierr = VecView(x, PETSC_VIEWER_DRAW_WORLD); CHKERRQ(ierr);
  ierr = RestoreGhostCells(mesh, numGhostCells); CHKERRQ(ierr);

  ierr = VecNorm(x, NORM_INFINITY, &xnorm); CHKERRQ(ierr);
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
