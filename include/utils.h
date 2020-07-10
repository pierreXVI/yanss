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
enum BCType {BC_DIRICHLET, BC_OUTFLOW_P, BC_WALL, BC_PERIO};


enum ProblemType {TYPE_EULER, TYPE_NS};


#define DOF_1   -1
#define DOF_DIM -2


typedef struct _Physics *Physics;
struct _Physics {
  enum ProblemType     type;    // Problem type

  PetscInt             dof;     // Total number of dof
  PetscInt             dim;     // Spatial dimention

  PetscReal            gamma;   // Heat capacity ratio

  struct BCDescription *bc;     // Boundary conditions
  struct BCCtx         *bc_ctx; // Boundary condition contexts
  PetscInt             nbc;     // Number of boundary conditions

  PetscReal            *init;   // Initial conditions, in primitive variables

  Vec                  x;       // Physical state
};

struct BCDescription {
  const char  *name; // Boundary name
  enum BCType type;  // Boundary type
  PetscReal   *val;  // Additional numerical values
};

struct BCCtx {
  Physics  phys; // Physical model
  PetscInt i;    // Boundary number
};

struct FieldDescription {
  const char *name;
  PetscInt   dof;
};

#endif
