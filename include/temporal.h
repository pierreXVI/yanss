#ifndef TEMPORAL
#define TEMPORAL

#include "utils.h"

/*
  Setup the PETSc Time-Stepper
  The `TS` must be freed with `TSDestroy`
*/
PetscErrorCode MyTsCreate(MPI_Comm, TS*, DM, Physics, PetscReal);

#endif
