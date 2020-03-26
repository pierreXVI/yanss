#ifndef STRUCTURES
#define STRUCTURES

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>

/* Represents continuum physical equations. */
typedef struct _n_Physics *Physics;

struct FieldDescription {
  const char *name;
  PetscInt dof;
};

struct _n_Physics {
  PetscRiemannFunc riemann;
  PetscInt         dof;          /* number of degrees of freedom per cell */
  PetscReal        maxspeed;     /* kludge to pick initial time step, need to add monitoring and step control */
  PetscReal        inflowState;
  PetscReal        wind[2];
  PetscInt         nfields;
  const struct FieldDescription *field_desc;
};


#endif
