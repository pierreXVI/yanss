#ifndef SPATIAL
#define SPATIAL

#include "structures.h"

PetscErrorCode SetMesh(MPI_Comm, DM*, PetscFV*, Physics);
/*
  Setup the mesh
  Allocate the DM and the PetscFV
*/

PetscErrorCode GetInitialCondition(DM, Vec, Physics);


#endif
