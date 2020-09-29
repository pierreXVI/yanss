#ifndef INCLUDE_FLAG_SPATIAL
#define INCLUDE_FLAG_SPATIAL

#include "utils.h"


/*
  A mesh is a `DMPlex` instance, with FV adjacency and a single `PetscFV` field
*/

typedef struct {
  PetscInt        n_perio; // Number of periodic BC
  struct PerioCtx *perio;  // Periodicity context
} *MeshCtx;

struct PerioCtx {
  Vec buffer;        // Buffer vector
  IS  master, slave; // Master and Slave cell ids
};


/*
  Destroy the mesh
*/
PetscErrorCode MeshDestroy(DM*);

/*
  Setup the mesh.
  The output must be freed with `MeshDestroy`.
  The mesh can be viewed with the options:
   -mesh_view_orig - To view the raw mesh from the input file
   -mesh_view      - To view the mesh produced by this function
*/
PetscErrorCode MeshLoadFromFile(MPI_Comm, const char*, const char*, DM*);


/*
  Apply a function on a mesh
*/
PetscErrorCode MeshApplyFunction(DM, PetscReal, PetscErrorCode(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*), void*, Vec);


/*
  Create a global vector, and set the user's Viewer
*/
PetscErrorCode MeshCreateGlobalVector(DM, Vec*);


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
  The Vec is linked to a mesh, so that the number of field components is read from the DM's PetscFV
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
  Sets up the mesh and the physical context (for the boundaries conditions)
*/
PetscErrorCode MeshSetUp(DM dm, Physics phys, const char *filename);


/*
  Puts coefficients which represent periodic values into the local solution vector
*/
PetscErrorCode MeshInsertPeriodicValues(DM dm, Vec locX);


/*
  Compute the gradient (global vector) of the given vector (local vector) using the precomputed data
*/
PetscErrorCode MeshReconstructGradientsFVM(DM, Vec, Vec);

#endif
