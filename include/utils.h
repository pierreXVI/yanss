#ifndef INCLUDE_FLAG_UTILS
#define INCLUDE_FLAG_UTILS

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>


/*
  The boudrary condition types. For each type correspond a value array:
    BC_DIRICHLET: the conservative field values
    BC_OUTFLOW_P: the pressure value
    BC_WALL:      no value
*/
enum BCType {BC_DIRICHLET, BC_OUTFLOW_P, BC_WALL};


enum ProblemType {TYPE_EULER, TYPE_NS};


#define DOF_1   -1
#define DOF_DIM -2


struct RiemannCtx {
  void (*riemann_solver)(PetscInt, PetscInt, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, PetscInt, const PetscScalar*, PetscScalar*, void*);

  void (*pressure_solver)(PetscInt, void*, PetscReal*, PetscReal*,                            // Dimension, physical context, primitive states
                          PetscReal, PetscReal, PetscReal*, PetscReal*, PetscReal, PetscReal, // Normal and tangential speeds, sound speeds
                          PetscReal*, PetscReal*);                                            // Output: intermediate pressure and speed
  PetscReal pressure_solver_eps;   // Epsilon on the pressure solver, triggers linearisation of the rarefaction formula
  PetscReal pressure_solver_niter; // Maximum number of iterations for the pressure solver
  PetscReal pressure_solver_rtol;  // Relative tolerance that triggers convergence of the pressure solver

  PetscReal advection_speed;

  PetscReal q_user;

};

typedef struct {
  enum ProblemType     type;    // Problem type

  PetscInt             dof;     // Total number of dof
  PetscInt             dim;     // Spatial dimention

  PetscReal            gamma;   // Heat capacity ratio

  struct BCCtx         *bc_ctx; // Boundary condition contexts
  PetscInt             nbc;     // Number of boundary conditions

  PetscReal            *init;   // Initial conditions, in primitive variables

  struct RiemannCtx    riemann_ctx; // Riemann solver context

  Vec                  x;       // Physical state
} *Physics;

struct BCCtx {
  Physics     phys;  // Physical model
  const char  *name; // Boundary name
  enum BCType type;  // Boundary type
  PetscReal   *val;  // Additional numerical values
};


typedef struct {
  DM              dm;      // DMPLEX object
  PetscInt        n_perio; // Number of periodic BC
  struct PerioCtx *perio;  // Periodicity context
} *Mesh;

struct PerioCtx {
  Vec buffer;        // Buffer vector
  IS  master, slave; // Master and Slave cell ids
};

#endif
