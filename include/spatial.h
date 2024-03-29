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
  This numbering corresponds to the local vectors, the global vectors local cells are the "true" mesh cells only.
  Be carefull with the PETSc ambiguation:
  `DMPlexGetGhostCellStratum` corresponds to the boundary cells, and the "ghost" `DMLabel` to the partition cells.

  The numerical flux is computed in `MeshComputeRHSFunctionFVM`:
    - the local conservative vector is converted to primitive
    - the primitive cell centered gradients are computed with `MeshReconstructGradientsFVM`
    - the primitive face centered fields are constructed with the cell centered gradients
    - the face flux is computed with the choosen Riemann solver
    - the flux are accumulated on each cell
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


/*
  Convert the field from conservative to primitive in each "leaf" of the PetscSF:
  this is called after a DMGlobalToLocalEnd.
  To get a local primitive vector, one must set the DM with
    `ierr = DMGlobalToLocalHookAdd(dm, NULL, GlobalConservativeToLocalPrimitive_Endhook, phys); CHKERRQ(ierr);`
  or call
    `ierr = GlobalConservativeToLocalPrimitive_Endhook(dm, NULL, INSERT_VALUES, locX, phys); CHKERRQ(ierr);`
  The first solution makes the ConservativeToPrimitive convertion automatic but mandatory. The second if more flexible.
*/
PetscErrorCode GlobalConservativeToLocalPrimitive_Endhook(DM, Vec, InsertMode, Vec, void*);

#endif
