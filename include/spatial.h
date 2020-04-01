#ifndef SPATIAL
#define SPATIAL

#include "utils.h"

PetscErrorCode MeshDestroy(DM*);
PetscErrorCode MeshLoad(MPI_Comm, const char*, DM*);
/*
  Setup the mesh
  Use MeshDestroy to free the memory
*/


PetscErrorCode MeshApplyFunction(DM dm, PetscReal time,
                                 PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*),
                                 void *ctx, Vec x);
/*
  Apply a function on a DM
*/

#endif
