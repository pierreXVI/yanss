#include "utils.h"
#include <petsc/private/dmlabelimpl.h>
#include <petsc/private/isimpl.h>

PetscErrorCode HideGhostCells(DM dm, PetscInt *n){
  PetscErrorCode ierr;
  DMLabel        label;
  PetscInt       dim, pHyb, pEnd;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);                    CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, dim, NULL, &pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &pHyb, NULL);  CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &label);             CHKERRQ(ierr);
  label->points[dim]->max -= (pEnd - pHyb);
  if (n) *n = (pEnd - pHyb);
  PetscFunctionReturn(0);
}

PetscErrorCode RestoreGhostCells(DM dm, PetscInt n){
  PetscErrorCode ierr;
  DMLabel        label;
  PetscInt       dim;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);        CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &label); CHKERRQ(ierr);
  label->points[dim]->max += n;
  PetscFunctionReturn(0);
}
