#ifndef INCLUDE_FLAG_PHYSICS
#define INCLUDE_FLAG_PHYSICS

#include "utils.h"

/*
  Destroy the physical model
*/
PetscErrorCode PhysicsDestroy(Physics*);

/*
  Create the physical model
  The output must be freed with `PhysicsDestroy`
*/
PetscErrorCode PhysicsCreate(Physics*, const char*, Mesh);

/*
  Apply the initial condition
*/
PetscErrorCode InitialCondition(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscReal*, void*);


/*
  Convert the variable between primitive and conservative.
  It is safe to call it with `in` == `out`
*/
void PrimitiveToConservative(Physics, const PetscReal*, PetscReal*);
void ConservativeToPrimitive(Physics, const PetscReal*, PetscReal*);


/*
  Compute physical value from components
  ctx is to be casted to (Physics)
*/
PetscErrorCode normU(PetscInt Nc, const PetscReal *x, PetscScalar *y, void *ctx);
PetscErrorCode mach (PetscInt Nc, const PetscReal *x, PetscScalar *y, void *ctx);

/*
  Register the Riemann solvers in the given PetscFunctionList
    "advection", constant advection, for debugging purposes
    "exact",     exact Riemann solver
    "lax",       Lax Friedrich Riemann solver
    "anrs",      Adaptative Noniterative Riemann Solver
*/
PetscErrorCode Register_RiemannSolver(PetscFunctionList*);


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
PetscErrorCode BCDirichlet(PetscReal, const PetscReal[3], const PetscReal[3], const PetscReal*, PetscReal*, void*);
PetscErrorCode BCOutflow_P(PetscReal, const PetscReal[3], const PetscReal[3], const PetscReal*, PetscReal*, void*);
PetscErrorCode BCWall     (PetscReal, const PetscReal[3], const PetscReal[3], const PetscReal*, PetscReal*, void*);

#endif
