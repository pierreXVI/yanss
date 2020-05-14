#ifndef INCLUDE_FLAG_TEMPORAL
#define INCLUDE_FLAG_TEMPORAL

#include "utils.h"

/*
  Setup the PETSc Time-Stepper
  The `TS` must be freed with `TSDestroy`
*/
PetscErrorCode MyTsCreate(MPI_Comm, TS*, const char*, DM, Physics, PetscReal);

#endif
