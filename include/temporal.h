#ifndef TEMPORAL
#define TEMPORAL

#include "utils.h"


PetscErrorCode MyTsCreate(MPI_Comm, TS*, DM, PetscReal);
/*
  Setup the PETSc Time-Stepper
  Allocate a TS
*/


#endif
