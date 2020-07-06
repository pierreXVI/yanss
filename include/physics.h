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
PetscErrorCode PhysicsCreate(Physics*, const char*, DM);

/*
  Apply the initial condition
*/
PetscErrorCode InitialCondition(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*);


/*
  Convert the variable between primitive and conservative.
  It is safe to call it with `in` == `out`
*/
void PrimitiveToConservative(Physics, const PetscReal*, PetscReal*);
void ConservativeToPrimitive(Physics, const PetscReal*, PetscReal*);


/*
  Pointwise Riemann solver functions, with the following calling sequence:

  ```
  func(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[],
       const PetscScalar uR[], PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx)
    dim          - Spatial dimension
    Nf           - Number of fields
    x            - Coordinates at a point on the interface
    n            - Area-scaled normal vector to the interface
    uL           - State vector to the left of the interface
    uR           - State vector to the right of the interface
    flux         - Output array of flux through the interface
    numConstants - Number of constant parameters
    constants    - Constant parameters
    ctx          - Context, to be casted to (Physics)
  ```
*/
void RiemannSolver_Euler_Exact         (PetscInt, PetscInt, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, PetscInt, const PetscScalar*, PetscScalar*, void*);
void RiemannSolver_Euler_LaxFriedrichs (PetscInt, PetscInt, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, PetscInt, const PetscScalar*, PetscScalar*, void*);

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
PetscErrorCode BCFarField (PetscReal, const PetscReal[3], const PetscReal[3], const PetscScalar*, PetscScalar*, void*);

#endif
