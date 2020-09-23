#ifndef INCLUDE_FLAG_TEMPORAL
#define INCLUDE_FLAG_TEMPORAL

#include "utils.h"


/*
  Setup the PETSc Time-Stepper
  The `TS` must be freed with `TSDestroy`
  The time step is constant, based on the initial CFL value
  The TS can be viewed with the option `-ts_view`
*/
PetscErrorCode MyTsCreate(MPI_Comm, TS*, const char*, Mesh, Physics, PetscReal);

#endif
