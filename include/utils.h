#ifndef UTILS
#define UTILS

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>

enum BC_TYPE {BC_NULL, BC_DIRICHLET, BC_OUTFLOW, BC_WALL};

#define DOF_NULL 0
#define DOF_1   -1
#define DOF_DIM -2

typedef struct _Physics *Physics;

struct FieldDescription {
  const char *name;
  PetscInt   dof;
};

struct BCDescription {
  const char   *name;
  enum BC_TYPE type;
  PetscReal    *val;
};


struct BC_ctx {
  Physics  phys;
  PetscInt i;
};


struct _Physics {
  struct FieldDescription *fields; // The physical phields
  PetscInt                nfields; // The number of physical phields
  PetscInt                dof;     // The total number of dof
  PetscReal               *c;      // The advection speed
  struct BCDescription    *bc;     // The boundary conditions
  struct BC_ctx           *bc_ctx; // The boundary condition contexts
  PetscInt                nbc;     // The number of boundary conditions
  PetscInt                dim;     // The spatial dimention
};


PetscErrorCode HideGhostCells(DM, PetscInt*);
PetscErrorCode RestoreGhostCells(DM, PetscInt);

#endif
