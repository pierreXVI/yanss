#ifndef PHYSICS
#define PHYSICS

#include "utils.h"

/*
  Destroy the physical model
*/
PetscErrorCode PhysicsDestroy(Physics*);

/*
  Create the physical model
  The output must be freed with `PhysicsDestroy`
*/
PetscErrorCode PhysicsCreate(Physics*, const char*, DM);

/*
  Apply the initial condition
*/
PetscErrorCode InitialCondition(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*);


void PrimitiveToConservative(Physics, const PetscReal*, PetscReal*);
void ConservativeToPrimitive(Physics, const PetscReal*, PetscReal*);


void RiemannSolver_Euler_Exact(PetscInt, PetscInt, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, PetscInt, const PetscScalar*, PetscScalar*, void*);
void RiemannSolver_Euler_Lax  (PetscInt, PetscInt, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, PetscInt, const PetscScalar*, PetscScalar*, void*);

/*
  Pointwise boundary condition functions, with the following calling sequence:

  ```
  func(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscScalar *xI, PetscScalar *xG, void *ctx)
    time - Current time
    c    - Location of centroid (quadrature point)
    n    - Area-scaled normals
    xI   - Value on the limit cell
    xG   - Value on the ghost cell
    ctx  - Context, to be casted to (struct BC_ctx*)
  ```
*/
PetscErrorCode BCDirichlet(PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);
PetscErrorCode BCOutflow_P(PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);
PetscErrorCode BCWall     (PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);

#endif
