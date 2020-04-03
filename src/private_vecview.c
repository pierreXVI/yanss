#include "private_vecview.h"

#include <petsc/private/vecimpl.h>
#include <petscdraw.h>

PETSC_EXTERN PetscErrorCode VecView_Plex(Vec, PetscViewer);

/*
    Allows a user to select a subset of the field components to be drawn by VecView() when the vector comes from a DMPlex with one field
    Ispired from DMDASelectFields
*/
PetscErrorCode PetscSectionSelectFieldComponents(PetscSection s, PetscInt f, PetscInt *outcomponents, PetscInt **components)
{
  PetscErrorCode ierr;
  PetscInt       Nc, ndisplaycomp, *displaycomp, k;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscSectionGetFieldComponents(s, f, &Nc); CHKERRQ(ierr);
  ierr = PetscMalloc1(Nc, &displaycomp); CHKERRQ(ierr);
  for (k = 0; k < Nc; k++) displaycomp[k] = k;
  ndisplaycomp = Nc;
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-draw_comp", displaycomp, &ndisplaycomp, &flg); CHKERRQ(ierr);
  if (!ndisplaycomp) ndisplaycomp = Nc;
  *components    = displaycomp;
  *outcomponents = ndisplaycomp;
  PetscFunctionReturn(0);
}

static PetscErrorCode MyVecView_Plex_Local_Draw(Vec v, PetscViewer viewer)
{
  DM                 dm;
  PetscSection       s;
  PetscDraw          draw, popup;
  DM                 cdm;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *array;
  PetscReal          bound[4] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscReal          vbound[2], time;
  PetscBool          isnull, flg;
  PetscInt           dim, Nf, Nc, comp, vStart, vEnd, cStart, cEnd, c, N, level, step;
  const char        *name;
  char               title[PETSC_MAX_PATH_LEN];
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer, 0, &draw); CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw, &isnull); CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = VecGetDM(v, &dm); CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dim); CHKERRQ(ierr);
  if (dim != 2) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes of dimension %D. Use PETSCVIEWERGLVIS", dim);
  ierr = DMGetLocalSection(dm, &s); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &Nf); CHKERRQ(ierr);
  if (Nf > 1) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes with multiple fields (%d).", Nf);
  ierr = DMGetCoarsenLevel(dm, &level); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm); CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection); CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);

  ierr = PetscObjectGetName((PetscObject) v, &name); CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &step, &time); CHKERRQ(ierr);

  ierr = VecGetLocalSize(coordinates, &N); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords); CHKERRQ(ierr);
  for (c = 0; c < N; c += dim) {
    bound[0] = PetscMin(bound[0], PetscRealPart(coords[c]));   bound[2] = PetscMax(bound[2], PetscRealPart(coords[c]));
    bound[1] = PetscMin(bound[1], PetscRealPart(coords[c+1])); bound[3] = PetscMax(bound[3], PetscRealPart(coords[c+1]));
  }
  ierr = VecRestoreArrayRead(coordinates, &coords); CHKERRQ(ierr);
  ierr = PetscDrawClear(draw); CHKERRQ(ierr);

  DM   fdm = dm;
  Vec  fv  = v;
  char prefix[PETSC_MAX_PATH_LEN];
  PetscInt i, ndisplaycomp, *displaycomp;

  PetscFV fvm;
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);

  ierr = PetscSectionGetFieldComponents(s, 0, &Nc); CHKERRQ(ierr);
  ierr = PetscSectionSelectFieldComponents(s, 0, &ndisplaycomp, &displaycomp); CHKERRQ(ierr);

  PetscInt  nmax = 2 * ndisplaycomp;
  PetscReal vbound_tot[nmax];
  ierr = PetscOptionsGetRealArray(NULL, prefix, "-vec_view_bounds", vbound_tot, &nmax, &flg); CHKERRQ(ierr);
  if (nmax < 2 * ndisplaycomp){
    for (PetscInt i = nmax; i < 2 * ndisplaycomp; i++){
      vbound_tot[i] = vbound_tot[i%2];
    }
  }

  if (v->hdr.prefix) {ierr = PetscStrncpy(prefix, v->hdr.prefix, sizeof(prefix)); CHKERRQ(ierr);}
  else               {prefix[0] = '\0';}
  for (i = 0; i < ndisplaycomp; ++i) {
    comp = displaycomp[i];
    const char *compName;
    ierr = PetscFVGetComponentName(fvm, comp, &compName); CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer, displaycomp[i], &draw); CHKERRQ(ierr);
    ierr = PetscSNPrintf(title, sizeof(title), "%s:%s Step: %D Time: %.4g", name, compName, step, time); CHKERRQ(ierr);
    ierr = PetscDrawSetTitle(draw, title); CHKERRQ(ierr);
    if (flg) {
      vbound[0] = vbound_tot[2*i];
      vbound[1] = vbound_tot[2*i + 1];
    } else {
      Vec subfv;
      IS  is;
      ierr = ISCreateStride(PetscObjectComm((PetscObject) dm), cEnd - cStart, comp, Nc, &is); CHKERRQ(ierr);
      ierr = VecGetSubVector(fv, is, &subfv); CHKERRQ(ierr);
      ierr = VecMin(subfv, NULL, &vbound[0]); CHKERRQ(ierr);
      ierr = VecMax(subfv, NULL, &vbound[1]); CHKERRQ(ierr);
      ierr = VecDestroy(&subfv); CHKERRQ(ierr);
      ierr = ISDestroy(&is); CHKERRQ(ierr);
      if (vbound[1] <= vbound[0]) vbound[1] = vbound[0] + 1.0;
    }
    ierr = PetscDrawGetPopup(draw, &popup); CHKERRQ(ierr);
    ierr = PetscDrawScalePopup(popup, vbound[0], vbound[1]); CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates(draw, bound[0], bound[1], bound[2], bound[3]); CHKERRQ(ierr);

    ierr = VecGetArrayRead(fv, &array); CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      PetscScalar *coords = NULL, *a = NULL;
      PetscInt     numCoords, color[4] = {-1,-1,-1,-1};

      ierr = DMPlexPointLocalRead(fdm, c, array, &a); CHKERRQ(ierr);
      if (a) {
        color[0] = PetscDrawRealToColor(PetscRealPart(a[comp]), vbound[0], vbound[1]);
        color[1] = color[2] = color[3] = color[0];
      } else {
        PetscScalar *vals = NULL;
        PetscInt     numVals, va;

        ierr = DMPlexVecGetClosure(fdm, NULL, fv, c, &numVals, &vals); CHKERRQ(ierr);
        if (numVals % Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of components %D does not divide the number of values in the closure %D", Nc, numVals);
        switch (numVals/Nc) {
        case 3: /* P1 Triangle */
        case 4: /* P1 Quadrangle */
          for (va = 0; va < numVals/Nc; ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va*Nc+comp]), vbound[0], vbound[1]);
          break;
        case 6: /* P2 Triangle */
        case 8: /* P2 Quadrangle */
          for (va = 0; va < numVals/(Nc*2); ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va*Nc+comp + numVals/(Nc*2)]), vbound[0], vbound[1]);
          break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of values for cell closure %D cannot be handled", numVals/Nc);
        }
        ierr = DMPlexVecRestoreClosure(fdm, NULL, fv, c, &numVals, &vals); CHKERRQ(ierr);
      }
      ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, &numCoords, &coords); CHKERRQ(ierr);
      switch (numCoords) {
      case 6:
        ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]); CHKERRQ(ierr);
        break;
      case 8:
        ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]); CHKERRQ(ierr);
        ierr = PetscDrawTriangle(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), color[2], color[3], color[0]); CHKERRQ(ierr);
        break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells with %D coordinates", numCoords);
      }
      ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, &numCoords, &coords); CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(fv, &array); CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw); CHKERRQ(ierr);
    if (i == ndisplaycomp - 1) {ierr = PetscDrawPause(draw); CHKERRQ(ierr);}
    ierr = PetscDrawSave(draw); CHKERRQ(ierr);
  }

  ierr = PetscFree(displaycomp); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
  Same as VecView_Plex, but when viewer is of type PETSCVIEWERDRAW,
  assumes the field classid is PETSCFV_CLASSID and uses MyVecView_Plex_Local_Draw
*/
PetscErrorCode MyVecView_Plex(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscBool      isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm); CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw); CHKERRQ(ierr);
  if (isdraw) {
    Vec        locv;
    const char *name;

    ierr = DMGetLocalVector(dm, &locv); CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) v, &name); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) locv, name); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv); CHKERRQ(ierr);
    ierr = MyVecView_Plex_Local_Draw(locv, viewer); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locv); CHKERRQ(ierr);
  } else {
    ierr = VecView_Plex(v, viewer); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
