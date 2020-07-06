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
  Same as VecView_Plex, but when viewer is of type PETSCVIEWERDRAW,
  assumes the field classid is PETSCFV_CLASSID and uses MyVecView_Plex_Local_Draw
*/
PetscErrorCode MyVecView_Plex(Vec, PetscViewer);


#endif
