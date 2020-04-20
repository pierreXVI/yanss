#ifndef SPATIAL
#define SPATIAL

#include "utils.h"

/*
  Destroy the mesh
*/
PetscErrorCode MeshDestroy(DM*);

/*
  Setup the mesh.
  The output must be freed with `MeshDestroy`
*/
PetscErrorCode MeshLoadFromFile(MPI_Comm, const char*, DM*);


/*
  Apply a function on a mesh
*/
PetscErrorCode MeshApplyFunction(DM dm, PetscReal time,
                                 PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*),
                                 void *ctx, Vec x);


/*
  Create a global vector, and set the user's Viewer
*/
PetscErrorCode MeshCreateGlobalVector(DM, Vec*);


#endif
