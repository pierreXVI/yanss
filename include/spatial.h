#ifndef INCLUDE_FLAG_SPATIAL
#define INCLUDE_FLAG_SPATIAL

#include "utils.h"


/*
  A mesh is a `DMPlex` instance, with FV adjacency and a single `PetscFV` field
*/


/*
  Destroy the mesh
*/
PetscErrorCode MeshDestroy(Mesh*);

/*
  Setup the mesh.
  The output must be freed with `MeshDestroy`.
  The mesh can be viewed with the options:
   -mesh_view_orig - To view the raw mesh from the input file
   -mesh_view      - To view the mesh produced by this function
*/
PetscErrorCode MeshLoadFromFile(MPI_Comm, const char*, const char*, Mesh*);


/*
  Apply a function on a mesh
*/
PetscErrorCode MeshDMApplyFunction(DM, PetscReal, PetscErrorCode(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*), void*, Vec);


/*
  Create a global vector, and set the user's Viewer
*/
PetscErrorCode MeshDMCreateGlobalVector(DM, Vec*);


/*
  Get the component vectors of x
  If Nc is not NULL, returns the number of fields
*/
PetscErrorCode VecGetComponentVectors(Vec, PetscInt*, Vec**);

/*
  Destroys the component vectors
*/
PetscErrorCode VecDestroyComponentVectors(Vec, Vec**);


/*
  Apply a pointwise function to a Vec
  The Vec is linked to a Mesh, so that the number of field components is read from the DM's PetscFV
  The pointwise function calling sequence is
  ```
  func(PetscInt Nc, const PetscScalar x[], PetscScalar *y, void *ctx)
    Nc           - Number of field components
    x            - Field value
    y            - Output scalar value
    ctx          - Optional context
  ```
*/
PetscErrorCode VecApplyFunctionComponents(Vec, Vec*, PetscErrorCode(PetscInt, const PetscScalar*, PetscScalar*, void*), void*);


/*
  Reads the periodicity from the input file, and construct the `perio` context array
  The periodicity contexts can only be created after some of the physical context is filled, as the number of components is needed
*/
PetscErrorCode MeshSetPeriodicity(Mesh, const char*);


/*
  Compute the RHS
  Fills the periodic boundaries first, using mesh->periodic
  ctx is to be casted to (Mesh)
*/
PetscErrorCode MeshDMTSComputeRHSFunctionFVM(DM, PetscReal, Vec, Vec, void*);


#endif
