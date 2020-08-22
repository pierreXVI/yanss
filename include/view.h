#ifndef INCLUDE_FLAG_VIEW
#define INCLUDE_FLAG_VIEW

#include <petscdmplex.h>
#include <petscdraw.h>


/*
  View a Mesh based vector
  Calls VecView_Mesh_Local_Draw if the viewer is of type PETSCVIEWERDRAW, else calls VecViex_Plex
*/
PetscErrorCode VecView_Mesh(Vec, PetscViewer);


/*
  Sets MeshDMView as the default object viewer for the given DM
*/
PetscErrorCode MeshDMSetViewer(DM dm);


#endif
