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
  Setup the Riemann solver from options:

  -riemann type, where type is:
    "advection", constant advection, for debugging purposes
    "exact",     exact Riemann solver (default)
    "lax",       Lax Friedrich Riemann solver
    "anrs",      Adaptive Noniterative Riemann Solver
    "roe",       Roe-Pike Riemann Solver

  -riemann_advection_speed value

  -riemann_exact_p_solver solver where solver is:
    "newton", Newton iteration solver (default)
    "fp",     Fixed-point iteration solver

  -riemann_exact_p_solver_rtol value, relative tolerance for the pressure solver
  -riemann_exact_p_solver_niter value, maximum number of iterations for the pressure solver
  -riemann_exact_p_solver_eps value, epsilon on the pressure solver, triggers linearisation of the rarefaction formula

  -riemann_anrs_q value, pressure ratio over which the PVRS is not used

  -riemann_roe_entropy_fix fix, where fix is:
    "none", no entropy fix (default)
    "hh1",  Harten-Hyman first entropy fix
    "hh2",  Harten-Hyman second entropy fix
*/
PetscErrorCode PhysicsRiemannSetFromOptions(MPI_Comm,
                                            void (**)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void*),
                                            union RiemannCtx*);

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
PetscErrorCode BCDirichlet(PetscReal, const PetscReal[], const PetscReal[], const PetscReal*, PetscReal*, void*);
PetscErrorCode BCOutflow_P(PetscReal, const PetscReal[], const PetscReal[], const PetscReal*, PetscReal*, void*);
PetscErrorCode BCWall     (PetscReal, const PetscReal[], const PetscReal[], const PetscReal*, PetscReal*, void*);

#endif
