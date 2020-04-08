#ifndef UTILS
#define UTILS

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>

enum BCType {BC_NULL, BC_DIRICHLET, BC_OUTFLOW_P, BC_WALL};
PETSC_EXTERN const char * const BCTypes[];

enum ProblemType {TYPE_EULER, TYPE_NS};
PETSC_EXTERN const char * const ProblemTypes[];


#define DOF_NULL 0
#define DOF_1   -1
#define DOF_DIM -2

typedef struct _Physics *Physics;

struct FieldDescription {
  const char *name;
  PetscInt   dof;
};

struct BCDescription {
  const char        *name; // The boundary name
  const enum BCType type;  // The boundary type
  const PetscReal   *val;  // Additional numerical values
};

struct BC_ctx {
  Physics  phys; // The physical model
  PetscInt i;    // The boundary number
};

struct _Physics {
  enum ProblemType        type;    // The problem type
  struct FieldDescription *fields; // The physical phields
  PetscInt                nfields; // The number of physical phields
  PetscInt                dof;     // The total number of dof
  struct BCDescription    *bc;     // The boundary conditions
  struct BC_ctx           *bc_ctx; // The boundary condition contexts
  PetscInt                nbc;     // The number of boundary conditions
  PetscInt                dim;     // The spatial dimention
  PetscReal               gamma;   // The heat capacity ratio
};


PetscErrorCode HideGhostCells(DM, PetscInt*);
PetscErrorCode RestoreGhostCells(DM, PetscInt);

#endif
