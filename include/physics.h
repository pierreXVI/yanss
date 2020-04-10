#ifndef PHYSICS
#define PHYSICS

#include "utils.h"

// Number of iterations of the fixed point pressure solver for the Riemann problem
#define N_ITER_RIEMANN 10
// Epsilon of the pressure solver for the Riemann problem
#define EPS_RIEMANN 1E-14


#define RHO_0 1
#define U_0   1
#define U_1   1.1
#define P_0   1E5

PetscErrorCode PhysicsDestroy(Physics*);
PetscErrorCode PhysicsCreate(Physics*, DM);
/*
  Creates the physical model
  Use PhysicsDestroy to free the memory
*/

PetscErrorCode PhysSetupBC(Physics, PetscDS, struct BCDescription*);
/*
  Load the BC
  Register the BC in the PetscDS
  Pre-process them if needed
*/


PetscErrorCode InitialCondition(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*);

PetscErrorCode PrimitiveToConservative(Physics, const PetscReal*, PetscReal*);
PetscErrorCode ConservativeToPrimitive(Physics, const PetscReal*, PetscReal*);


void RiemannSolver_Euler_Exact(PetscInt, PetscInt, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, PetscInt, const PetscScalar*, PetscScalar*, void*);

PetscErrorCode BCDirichlet(PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);
PetscErrorCode BCOutflow_P(PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);
PetscErrorCode BCWall     (PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);

#endif
