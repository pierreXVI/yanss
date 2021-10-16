#ifndef INCLUDE_FLAG_SPATIAL
#define INCLUDE_FLAG_SPATIAL

#include "utils.h"


/*
  A mesh is a `DMPlex` instance, with FV adjacency and a single `PetscFV` field

  The mesh viewer is replaced when viewed from options: PetscObjectView uses MeshView but DMView is the same.
  The global vector viewer is also replaced with VecView_Mesh.

  The mesh contains three cell types:
    - the "true" mesh cells, in [cStartCell, cStartOverlap[
    - the overlapping cells in [cStartOverlap, cStartBoundary[, due to partitioning
    - the boundary cells in [cStartBoundary, cEnd[
  Be carefull with the PETSc ambiguation:
  `DMPlexGetGhostCellStratum` corresponds to the boundary cells, and the "ghost" `DMLabel` to the partition cells.
*/

typedef struct {
  PetscInt cStartCell; // The first "true" cell
  PetscInt cStartOverlap; // The first partition cell
  PetscInt cStartBoundary; // The first boundary cell
  PetscInt cEnd; // The upper bound on cells

  struct {
    IS        neighborhood; // List of neighbors
    PetscReal *grad_coeff;  // Neighbor contributions to cell gradient
  } *CellCtx;

  PetscReal *FaceCtx;

} *MeshCtx;


/*
  Destroy the mesh
*/
PetscErrorCode MeshDestroy(DM*);

/*
  Setup the mesh
  The output must be freed with `MeshDestroy`.
  The mesh can be viewed with the options:
   -mesh_view_orig - To view the raw mesh from the input file
   -mesh_view      - To view the mesh produced by this function
*/
PetscErrorCode MeshLoadFromFile(MPI_Comm, const char*, const char*, DM*);


/*
  Get the bounds cStartCell, cStartOverlap, cStartBoundary and cEnd that describe the mesh cells.
*/
PetscErrorCode MeshGetCellStratum(DM, PetscInt*, PetscInt*, PetscInt*, PetscInt*);

/*
  Apply a function on a mesh
*/
PetscErrorCode MeshApplyFunction(DM, PetscReal, PetscErrorCode(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscReal*, void*), void*, Vec);

/*
  Create a global vector, and set the correct viewer
*/
PetscErrorCode MeshCreateGlobalVector(DM, Vec*);

/*
  Sets up the mesh and the physical context (for the boundaries conditions)
*/
PetscErrorCode MeshSetUp(DM dm, Physics phys, const char *filename);

/*
  Compute the gradient (global vector) of the given vector (local vector) using the precomputed data
*/
PetscErrorCode MeshReconstructGradientsFVM(DM, Vec, Vec);


/*
  Apply an function to a Vec
  The number of field components is read from the block size
  The pointwise function calling sequence is
  ```
  func(const PetscReal x[], PetscReal y[], void *ctx)
    x            - Field value
    y            - Output scalar value
    ctx          - Optional context
  ```
*/
PetscErrorCode VecApplyFunctionInPlace(Vec, void(const PetscReal*, PetscReal*, void*), void*);

/*
  Create a Vec as the image of a pointwise function
  The number of field components is read from the block size
  The pointwise function calling sequence is
  ```
  func(const PetscReal x[], PetscReal *y, void *ctx)
    x            - Field value
    y            - Output scalar value
    ctx          - Optional context
  ```
*/
PetscErrorCode VecApplyFunctionComponents(Vec, Vec*, void(const PetscReal*, PetscReal*, void*), void*);

#endif
