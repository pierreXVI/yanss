#ifndef TEMPORAL
#define TEMPORAL

#include "structures.h"


PetscErrorCode SetTs(MPI_Comm, TS*, DM, PetscReal);
/*
  Setup the PETSc Time-Stepper
  Allocate a TS
*/


#endif
