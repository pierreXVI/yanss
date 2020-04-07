#include "temporal.h"

static PetscErrorCode TSMonitorAscii(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode ierr;
  PetscReal      xnorm;

  PetscFunctionBeginUser;
  ierr = VecNorm(u, NORM_INFINITY, &xnorm);                                                  CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "% 3D  time %8.4g  |x| %8.4g\n", steps, time, xnorm); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSMonitorDraw(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // PetscBool flg;
  // PetscInt  nmax = 2;
  // PetscReal vbound[nmax];
  // ierr = PetscOptionsGetRealArray(NULL, NULL, "-vec_view_bounds", vbound, &nmax, &flg); CHKERRQ(ierr);
  // if (!flg){
  //   static PetscInt buffer_size = 32;
  //   char            buffer[buffer_size];
  //   ierr = VecMin(u, NULL, &vbound[0]);                                                 CHKERRQ(ierr);
  //   ierr = VecMax(u, NULL, &vbound[1]);                                                 CHKERRQ(ierr);
  //   if (vbound[1] <= vbound[0]) vbound[1] = vbound[0] + 1.0;
  //   ierr = PetscSNPrintf(buffer, buffer_size, "%f,%f", vbound[0], vbound[1]);           CHKERRQ(ierr);
  //   ierr = PetscOptionsSetValue(NULL, "-vec_view_bounds", buffer);                      CHKERRQ(ierr);
  // }

  DM       dm;
  PetscInt numGhostCells;
  ierr = VecGetDM(u, &dm);                     CHKERRQ(ierr);
  ierr = HideGhostCells(dm, &numGhostCells);   CHKERRQ(ierr);
  ierr = VecView(u, PETSC_VIEWER_DRAW_WORLD);  CHKERRQ(ierr);
  ierr = RestoreGhostCells(dm, numGhostCells); CHKERRQ(ierr);

  // if (!flg){
  //   ierr = PetscOptionsClearValue(NULL, "-vec_view_bounds");                            CHKERRQ(ierr);
  // }

  PetscFunctionReturn(0);
}


PetscErrorCode MyTsCreate(MPI_Comm comm, TS *ts, DM dm, Physics phys, PetscReal dt){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSCreate(comm, ts);                                   CHKERRQ(ierr);
  ierr = TSSetDM(*ts, dm);                                     CHKERRQ(ierr);
  ierr = TSSetTimeStep(*ts, dt);                               CHKERRQ(ierr);
  ierr = TSSetType(*ts, TSEULER);                              CHKERRQ(ierr);
  ierr = TSSetMaxTime(*ts, 2.0);                               CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSMonitorSet(*ts, TSMonitorAscii, NULL, NULL);        CHKERRQ(ierr);
  ierr = TSMonitorSet(*ts, TSMonitorDraw, phys, NULL);         CHKERRQ(ierr);
  ierr = TSSetFromOptions(*ts);                                CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
