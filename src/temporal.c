#include "temporal.h"

PetscErrorCode TSMonitorAscii_MinMax(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%3d  time %8.4g  ", steps, time); CHKERRQ(ierr);

  DM       dm;
  PetscFV  fvm;
  PetscInt Nc, size;
  ierr = VecGetDM(u, &dm);                             CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);            CHKERRQ(ierr);
  ierr = VecGetLocalSize(u, &size);                    CHKERRQ(ierr);
  for (PetscInt comp = 0; comp < Nc; comp++) {
    IS         is;
    Vec        subv;
    PetscReal   min, max;
    const char *compName;
    ierr = ISCreateStride(PetscObjectComm((PetscObject) u), size / Nc, comp, Nc, &is);     CHKERRQ(ierr);
    ierr = VecGetSubVector(u, is, &subv);                                                  CHKERRQ(ierr);
    ierr = VecMin(subv, PETSC_NULL, &min);                                                       CHKERRQ(ierr);
    ierr = VecMax(subv, PETSC_NULL, &max);                                                       CHKERRQ(ierr);
    ierr = VecDestroy(&subv);                                                              CHKERRQ(ierr);
    ierr = ISDestroy(&is);                                                                 CHKERRQ(ierr);
    ierr = PetscFVGetComponentName(fvm, comp, &compName);                                  CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s : [% 10.4g, % 10.4g], ", compName, min, max); CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\033[2D\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorAscii_Res(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (steps % 100 != 0) {PetscFunctionReturn(0);}

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%3d  time %8.4g  ", steps, time); CHKERRQ(ierr);

  Vec flux;
  ierr = VecDuplicate(u, &flux);                  CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts, time, u, flux); CHKERRQ(ierr);

  DM       dm;
  PetscFV  fvm;
  PetscInt Nc, size;
  ierr = VecGetDM(u, &dm);                             CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, PETSC_NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);            CHKERRQ(ierr);
  ierr = VecGetLocalSize(flux, &size);                 CHKERRQ(ierr);
  for (PetscInt comp = 0; comp < Nc; comp++) {
    IS         is;
    Vec        subv;
    PetscReal   min, max;
    const char *compName;
    ierr = ISCreateStride(PetscObjectComm((PetscObject) flux), size / Nc, comp, Nc, &is);  CHKERRQ(ierr);
    ierr = VecGetSubVector(flux, is, &subv);                                               CHKERRQ(ierr);
    ierr = VecMin(subv, PETSC_NULL, &min);                                                       CHKERRQ(ierr);
    ierr = VecMax(subv, PETSC_NULL, &max);                                                       CHKERRQ(ierr);
    ierr = VecDestroy(&subv);                                                              CHKERRQ(ierr);
    ierr = ISDestroy(&is);                                                                 CHKERRQ(ierr);
    ierr = PetscFVGetComponentName(fvm, comp, &compName);                                  CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s : % 10.4g, ", compName, PetscMax(-min, max)); CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\033[2D\n"); CHKERRQ(ierr);


  ierr = VecDestroy(&flux);

  // ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorDraw(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // PetscBool flg;
  // PetscInt  nmax = 2;
  // PetscReal vbound[nmax];
  // ierr = PetscOptionsGetRealArray(PETSC_NULL, PETSC_NULL, "-vec_view_bounds", vbound, &nmax, &flg); CHKERRQ(ierr);
  // if (!flg){
  //   static PetscInt buffer_size = 32;
  //   char            buffer[buffer_size];
  //   ierr = VecMin(u, PETSC_NULL, &vbound[0]);                                                 CHKERRQ(ierr);
  //   ierr = VecMax(u, PETSC_NULL, &vbound[1]);                                                 CHKERRQ(ierr);
  //   if (vbound[1] <= vbound[0]) vbound[1] = vbound[0] + 1.0;
  //   ierr = PetscSNPrintf(buffer, buffer_size, "%f,%f", vbound[0], vbound[1]);           CHKERRQ(ierr);
  //   ierr = PetscOptionsSetValue(PETSC_NULL, "-vec_view_bounds", buffer);                      CHKERRQ(ierr);
  // }

  DM       dm;
  PetscInt numGhostCells;
  ierr = VecGetDM(u, &dm);                           CHKERRQ(ierr);
  ierr = DMPlexHideGhostCells(dm, &numGhostCells);   CHKERRQ(ierr);
  ierr = VecView(u, PETSC_VIEWER_DRAW_WORLD);        CHKERRQ(ierr);
  ierr = DMPlexRestoreGhostCells(dm, numGhostCells); CHKERRQ(ierr);

  // if (!flg){
  //   ierr = PetscOptionsClearValue(PETSC_NULL, "-vec_view_bounds");                            CHKERRQ(ierr);
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
  // ierr = TSSetMaxTime(*ts, 0.001);                               CHKERRQ(ierr);
  ierr = TSSetMaxTime(*ts, 2);                               CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSMonitorSet(*ts, TSMonitorAscii_Res, PETSC_NULL, PETSC_NULL);    CHKERRQ(ierr);
  // ierr = TSMonitorSet(*ts, TSMonitorAscii_MinMax, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
  ierr = TSMonitorSet(*ts, TSMonitorDraw, PETSC_NULL, PETSC_NULL);         CHKERRQ(ierr);
  ierr = TSSetFromOptions(*ts);                                CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
