#ifndef INCLUDE_FLAG_PRIVATE_IMPL
#define INCLUDE_FLAG_PRIVATE_IMPL

#include <petscdmplex.h>
#include <petscdraw.h>

/*
  Hide and restore the ghost cells from a `DMPlex`
*/
PetscErrorCode DMPlexHideGhostCells(DM, PetscInt*);
PetscErrorCode DMPlexRestoreGhostCells(DM, PetscInt);


/*
  View a vector on a mesh
*/
PetscErrorCode VecView_Mesh(Vec, PetscViewer);



PetscErrorCode MeshSetViewer(DM dm);


#endif
