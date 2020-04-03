#include "private_vecview.h"


PetscErrorCode MyVecView_Plex(Vec v, PetscViewer viewer)
{
  // DM             dm;
  // PetscBool      isvtk, ishdf5, isdraw, isglvis;
  // PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "In MyVecView_Plex\n");
//   ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
//   if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
//   ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
//   ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
//   ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw);CHKERRQ(ierr);
//   ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
//   if (isvtk || isdraw || isglvis) {
//     Vec         locv;
//     const char *name;
//
//     ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
//     ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
//     ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
//     ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
//     ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
//     ierr = VecView_Plex_Local(locv, viewer);CHKERRQ(ierr);
//     ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
//   } else if (ishdf5) {
// #if defined(PETSC_HAVE_HDF5)
//     ierr = VecView_Plex_HDF5_Internal(v, viewer);CHKERRQ(ierr);
// #else
//     SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
// #endif
//   } else {
//     PetscBool isseq;
//
//     ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
//     if (isseq) {ierr = VecView_Seq(v, viewer);CHKERRQ(ierr);}
//     else       {ierr = VecView_MPI(v, viewer);CHKERRQ(ierr);}
//   }
  PetscFunctionReturn(0);
}
