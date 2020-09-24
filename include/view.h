#ifndef INCLUDE_FLAG_VIEW
#define INCLUDE_FLAG_VIEW

#include <petscdmplex.h>
#include <petscdraw.h>


/*
  View a mesh based vector
  Calls VecView_Mesh_Local_Draw if the viewer is of type PETSCVIEWERDRAW, else calls VecViex_Plex
  If so, the following options may be given:
    -draw_comp array,       select the vector components
    -vec_view_bounds array, specify the bounds for each vector components
    -vec_view_mesh,         view the mesh
    -vec_view_partition,    view the partition
*/
PetscErrorCode VecView_Mesh(Vec, PetscViewer);


/*
  Sets MeshView as the default object viewer for the given DM
*/
PetscErrorCode MeshSetViewer(DM dm);


#endif
