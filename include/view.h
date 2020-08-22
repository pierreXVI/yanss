#ifndef INCLUDE_FLAG_VIEW
#define INCLUDE_FLAG_VIEW

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



PetscErrorCode MeshDMSetViewer(DM dm);


#endif
