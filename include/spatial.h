#ifndef SPATIAL
#define SPATIAL

#include "utils.h"

/*
  A mesh is a `DMPlex` instance, with FV adjacency and a single `PetscFV` field
*/


/*
  Destroy the mesh
*/
PetscErrorCode MeshDestroy(DM*);

/*
  Setup the mesh.
  The output must be freed with `MeshDestroy`.
  The mesh can be inspected with the options:
   -dm_view_orig - To view the raw mesh from the input file
   -dm_view      - To view the mesh produced by this function
*/
PetscErrorCode MeshLoadFromFile(MPI_Comm, const char*, DM*);


/*
  Apply a function on a mesh
*/
PetscErrorCode MeshApplyFunction(DM, PetscReal, PetscErrorCode(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*), void*, Vec);


/*
  Create a global vector, and set the user's Viewer
*/
PetscErrorCode MeshCreateGlobalVector(DM, Vec*);


#endif
