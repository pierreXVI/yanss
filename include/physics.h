#ifndef PHYSICS
#define PHYSICS

#include "utils.h"

PetscErrorCode PhysicsDestroy(Physics*);
PetscErrorCode PhysicsCreate(Physics*, DM);
/*
  Creates the physical model
  Use PhysicsDestroy to free the memory
*/

PetscErrorCode InitialCondition(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*);

// void RiemannSolver(PetscInt, PetscInt, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, PetscInt, const PetscScalar*, PetscScalar*, void*);

// PetscErrorCode SetBC(DM, Physics);
//
//
//

#endif
