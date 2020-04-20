#ifndef UTILS
#define UTILS

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>

enum BCType {BC_DIRICHLET, BC_OUTFLOW_P, BC_WALL};

enum ProblemType {TYPE_EULER, TYPE_NS};


#define DOF_1   -1
#define DOF_DIM -2

typedef struct _Physics *Physics;

struct FieldDescription {
  const char *name;
  PetscInt   dof;
};

struct BCDescription {
  const char  *name; // Boundary name
  enum BCType type;  // Boundary type
  PetscReal   *val;  // Additional numerical values
};

struct BC_ctx {
  Physics  phys; // Physical model
  PetscInt i;    // Boundary number
};

struct _Physics {
  enum ProblemType        type;    // Problem type

  PetscInt                dof;     // Total number of dof
  PetscInt                dim;     // Spatial dimention

  PetscReal               gamma;   // Heat capacity ratio

  struct BCDescription    *bc;     // Boundary conditions
  struct BC_ctx           *bc_ctx; // Boundary condition contexts
  PetscInt                nbc;     // Number of boundary conditions

  PetscReal               *init;   // Initial conditions
};

/*
  Hide and restore the ghost cells from a `DMPlex`
*/
PetscErrorCode DMPlexHideGhostCells(DM, PetscInt*);
PetscErrorCode DMPlexRestoreGhostCells(DM, PetscInt);

/*
  Wrapping of `strdup`
*/
PetscErrorCode MyStrdup(const char*, const char**);

#endif
