#include "temporal.h"
#include "spatial.h"

static PetscErrorCode Monitor(TS ts, PetscInt stepnum, PetscReal time, Vec x, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscBool flg;
  PetscInt  nmax = 2;
  PetscReal vbound[nmax];
  ierr = PetscOptionsGetRealArray(NULL, NULL, "-vec_view_bounds", vbound, &nmax, &flg); CHKERRQ(ierr);
  if (!flg){
    static PetscInt buffer_size = 32;
    char buffer[buffer_size];
    ierr = VecMin(x, NULL, &vbound[0]);                                                 CHKERRQ(ierr);
    ierr = VecMax(x, NULL, &vbound[1]);                                                 CHKERRQ(ierr);
    if (vbound[1] <= vbound[0]) vbound[1] = vbound[0] + 1.0;
    ierr = PetscSNPrintf(buffer, buffer_size, "%f,%f", vbound[0], vbound[1]);           CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL, "-vec_view_bounds", buffer);                      CHKERRQ(ierr);
  }

  DM       dm;
  PetscInt numGhostCells;
  ierr = VecGetDM(x, &dm);                     CHKERRQ(ierr);
  ierr = HideGhostCells(dm, &numGhostCells);   CHKERRQ(ierr);
  ierr = VecView(x, PETSC_VIEWER_DRAW_WORLD);  CHKERRQ(ierr);
  ierr = RestoreGhostCells(dm, numGhostCells); CHKERRQ(ierr);

  if (!flg){
    ierr = PetscOptionsClearValue(NULL, "-vec_view_bounds"); CHKERRQ(ierr);
  }

  PetscReal xnorm;
  ierr = VecNorm(x, NORM_INFINITY, &xnorm);                               CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "% 3D  time %8.4g  |x| %8.4g\n", \
                     stepnum, (double)time, (double)xnorm);               CHKERRQ(ierr);
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
