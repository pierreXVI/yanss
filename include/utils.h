#ifndef INCLUDE_FLAG_UTILS
#define INCLUDE_FLAG_UTILS

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscsf.h>


#define DOF_1   -1
#define DOF_DIM -2

union RiemannCtx{
  struct {
    PetscReal advection_speed;
  }; // RiemannSolver_AdvectionX

  struct {
    void (*pressure_solver)(PetscInt, void*, const PetscReal*, const PetscReal*,                // Dimension, physical context, primitive states
                            PetscReal, PetscReal, PetscReal*, PetscReal*, PetscReal, PetscReal, // Normal and tangential speeds, sound speeds
                            PetscReal*, PetscReal*);                                            // Output: intermediate pressure and speed
    PetscReal pressure_solver_eps;   // Epsilon on the pressure solver, triggers linearisation of the rarefaction formula
    PetscReal pressure_solver_niter; // Maximum number of iterations for the pressure solver
    PetscReal pressure_solver_rtol;  // Relative tolerance that triggers convergence of the pressure solver
  }; // RiemannSolver_Exact

  struct {
    PetscReal q_user;
  }; // RiemannSolver_ANRS

  struct {
    void (*entropy_fix)(PetscReal*, PetscReal, PetscReal);
  }; // RiemannSolver_RoePike
};

typedef struct {
  PetscInt             dof;         // Total number of dof
  PetscInt             dim;         // Spatial dimention

  PetscReal            gamma;       // Heat capacity ratio
  PetscReal            r_gas;       // Specific gas constant (R / M)
  PetscReal            mu;          // Dynamic viscosity
  PetscReal            lambda;      // Thermal conductivity

  struct BCCtx         *bc_ctx;     // Boundary condition contexts
  PetscInt             nbc;         // Number of boundary conditions

  PetscReal            *init;       // Initial conditions, in primitive variables

  union RiemannCtx     riemann_ctx; // Riemann solver context
  void (*riemann_solver)(PetscInt, PetscInt,
                         const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[],
                         PetscInt, const PetscReal[], PetscReal[], void*);
} *Physics;

struct BCCtx {
  Physics     phys;  // Physical model
  const char  *name; // Boundary name
  const char  *type; // Boundary type
  PetscReal   *val;  // Additional numerical values
};

#endif
