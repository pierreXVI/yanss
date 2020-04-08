#ifndef PHYSICS
#define PHYSICS

#include "utils.h"

// Number of iterations of the fixed point pressure solver for the Riemann problem
#define N_ITER_RIEMANN 10

PetscErrorCode PhysicsDestroy(Physics*);
PetscErrorCode PhysicsCreate(Physics*, DM);
/*
  Creates the physical model
  Use PhysicsDestroy to free the memory
*/

PetscErrorCode InitialCondition(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*);


void RiemannSolver_Euler_Exact(PetscInt, PetscInt, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, PetscInt, const PetscScalar*, PetscScalar*, void*);

PetscErrorCode BCDirichlet(PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);
PetscErrorCode BCOutflow_P(PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);
PetscErrorCode BCWall     (PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);

#endif
