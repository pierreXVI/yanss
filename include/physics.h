#ifndef PHYSICS
#define PHYSICS

#include "utils.h"

#define N_ITER_RIEMANN 10

PetscErrorCode PhysicsDestroy(Physics*);
PetscErrorCode PhysicsCreate(Physics*, DM);
/*
  Creates the physical model
  Use PhysicsDestroy to free the memory
*/

PetscErrorCode InitialCondition(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*);

#endif
