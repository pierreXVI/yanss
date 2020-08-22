#include "view.h"


/*
    Allows a user to select a subset of the field components to be drawn by VecView() when the vector comes from a DMPlex with one field
    Ispired from DMDASelectFields
*/
static PetscErrorCode PetscSectionSelectFieldComponents(PetscSection s, PetscInt f, PetscInt *outcomponents, PetscInt **components){
  PetscErrorCode ierr;
  PetscInt       Nc, ndisplaycomp, *displaycomp, k;
  PetscBool      flg;

  PetscFunctionBeginUser;
  ierr = PetscSectionGetFieldComponents(s, f, &Nc); CHKERRQ(ierr);
  ierr = PetscMalloc1(Nc, &displaycomp);            CHKERRQ(ierr);
  for (k = 0; k < Nc; k++) displaycomp[k] = k;
  ndisplaycomp = Nc;
  ierr = PetscOptionsGetIntArray(PETSC_NULL, PETSC_NULL, "-draw_comp", displaycomp, &ndisplaycomp, &flg); CHKERRQ(ierr);
  if (!ndisplaycomp) ndisplaycomp = Nc;
  *components = displaycomp;
  *outcomponents = ndisplaycomp;
  PetscFunctionReturn(0);
}



#include <petsc/private/vecimpl.h>

PETSC_EXTERN PetscErrorCode VecView_Plex(Vec, PetscViewer);

static PetscErrorCode VecView_Mesh_Local_Draw(Vec v, PetscViewer viewer){
  PetscErrorCode     ierr;
  DM                 dm, cdm;
  PetscDraw          draw, popup;
  PetscSection       s, coordSection;
  Vec                coordinates;
  const PetscScalar  *coords, *array;
  PetscReal          bound[4] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL}, vbound[2], time;
  PetscBool          isnull, flg;
  PetscInt           dim, Nc, comp, cStart, cEnd, c, N, step;
  const char         *name;
  char               title[PETSC_MAX_PATH_LEN];
  DMLabel            ghostLabel;

  PetscFunctionBeginUser;
  ierr = PetscViewerDrawGetDraw(viewer, 0, &draw); CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw, &isnull);           CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = VecGetDM(v, &dm);             CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dim); CHKERRQ(ierr);
  if (dim != 2) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes of dimension %D.", dim);

  ierr = DMGetLabel(dm, "ghost", &ghostLabel);         CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);                  CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);        CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);      CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL); CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &cEnd, NULL);   CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &name);   CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &step, &time);  CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &N);             CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);        CHKERRQ(ierr);
  for (c = 0; c < N; c += dim) {
    bound[0] = PetscMin(bound[0], PetscRealPart(coords[c]));     bound[2] = PetscMax(bound[2], PetscRealPart(coords[c]));
    bound[1] = PetscMin(bound[1], PetscRealPart(coords[c + 1])); bound[3] = PetscMax(bound[3], PetscRealPart(coords[c + 1]));
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);                                                          CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, bound + 0, 2, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, bound + 2, 2, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);

  char     prefix[PETSC_MAX_PATH_LEN];
  PetscInt i, ndisplaycomp, *displaycomp;

  ierr = DMGetLocalSection(dm, &s);                                               CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(s, 0, &Nc);                               CHKERRQ(ierr);
  ierr = PetscSectionSelectFieldComponents(s, 0, &ndisplaycomp, &displaycomp);    CHKERRQ(ierr);
  if (v->hdr.prefix) {ierr = PetscStrncpy(prefix, v->hdr.prefix, sizeof(prefix)); CHKERRQ(ierr);}
  else               {prefix[0] = '\0';}

  PetscInt  nmax = 2 * ndisplaycomp;
  PetscReal vbound_tot[nmax];
  ierr = PetscOptionsGetRealArray(PETSC_NULL, prefix, "-vec_view_bounds", vbound_tot, &nmax, &flg); CHKERRQ(ierr);
  if (nmax < 2 * ndisplaycomp){
    for (PetscInt i = nmax; i < 2 * ndisplaycomp; i++) vbound_tot[i] = (i % 2) ? PETSC_MAX_REAL : PETSC_MIN_REAL;
  }



  for (i = 0; i < ndisplaycomp; ++i) {
    comp = displaycomp[i];
    const char *cname;
    ierr = PetscSectionGetComponentName(s, 0, comp, &cname);                                          CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer, i, &draw);                                                  CHKERRQ(ierr);
    ierr = PetscSNPrintf(title, sizeof(title), "%s:%s Step: %D Time: %.4g", name, cname, step, time); CHKERRQ(ierr);
    ierr = PetscDrawSetTitle(draw, title);                                                            CHKERRQ(ierr);
    if (flg) {
      vbound[0] = vbound_tot[2*i];
      vbound[1] = vbound_tot[2*i + 1];
    } else {
      vbound[0] = 0;
      vbound[1] = 0;
    }
    if (vbound[0] >= vbound[1]) {
      Vec subv;
      IS  is;
      ierr = ISCreateStride(PetscObjectComm((PetscObject) v), cEnd - cStart, comp, Nc, &is); CHKERRQ(ierr);
      ierr = VecGetSubVector(v, is, &subv);                                                  CHKERRQ(ierr);
      ierr = VecMin(subv, NULL, &vbound[0]);                                                 CHKERRQ(ierr);
      ierr = VecMax(subv, NULL, &vbound[1]);                                                 CHKERRQ(ierr);

      ierr = VecRestoreSubVector(v, is, &subv);                                              CHKERRQ(ierr);
      ierr = ISDestroy(&is);                                                                 CHKERRQ(ierr);
      if (vbound[1] <= vbound[0]) vbound[1] = vbound[0] + 1.0;
      ierr = MPIU_Allreduce(MPI_IN_PLACE, vbound + 0, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
      ierr = MPIU_Allreduce(MPI_IN_PLACE, vbound + 1, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
    }
    ierr = PetscDrawGetPopup(draw, &popup);                                       CHKERRQ(ierr);
    ierr = PetscDrawScalePopup(popup, vbound[0], vbound[1]);                      CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates(draw, bound[0], bound[1], bound[2], bound[3]); CHKERRQ(ierr);

    ierr = VecGetArrayRead(v, &array);                       CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      PetscInt ghostVal;
      ierr = DMLabelGetValue(ghostLabel, c, &ghostVal); CHKERRQ(ierr);
      if (ghostVal > 0) continue;

      PetscScalar    *coords = NULL, *a = NULL;
      PetscInt       numCoords, color[4];
      DMPolytopeType ct;

      ierr = DMPlexPointLocalRead(dm, c, array, &a); CHKERRQ(ierr);
      if (a) {
        color[1] = color[2] = color[3] = color[0] = PetscDrawRealToColor(PetscRealPart(a[comp]), vbound[0], vbound[1]);
      } else {
        PetscScalar *vals = PETSC_NULL;
        PetscInt     numVals, va;

        ierr = DMPlexVecGetClosure(dm, PETSC_NULL, v, c, &numVals, &vals); CHKERRQ(ierr);
        if (numVals % Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of components %D does not divide the number of values in the closure %D", Nc, numVals);
        switch (numVals/Nc) {
        case 3: /* P1 Triangle */
        case 4: /* P1 Quadrangle */
          for (va = 0; va < numVals/Nc; ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va*Nc + comp]), vbound[0], vbound[1]);
          break;
        case 6: /* P2 Triangle */
        case 8: /* P2 Quadrangle */
          for (va = 0; va < numVals / (2*Nc); ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va*Nc + comp + numVals / (2*Nc)]), vbound[0], vbound[1]);
          break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of values for cell closure %D cannot be handled", numVals/Nc);
        }
        ierr = DMPlexVecRestoreClosure(dm, PETSC_NULL, v, c, &numVals, &vals); CHKERRQ(ierr);
      }

      ierr = DMPlexGetCellType(dm, c, &ct);                                              CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, &numCoords, &coords); CHKERRQ(ierr);
      switch (ct) {
      case DM_POLYTOPE_TRIANGLE:
        ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]); CHKERRQ(ierr);
        // ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
        // ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
        // ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
        break;
      case DM_POLYTOPE_QUADRILATERAL:
        ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]); CHKERRQ(ierr);
        ierr = PetscDrawTriangle(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), color[2], color[3], color[0]); CHKERRQ(ierr);
        // ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
        // ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
        // ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
        // ierr = PetscDrawLine(draw, PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
        break;
      case DM_POLYTOPE_FV_GHOST:
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of type %s", DMPolytopeTypes[ct]);
        break;
      }
      ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, &numCoords, &coords); CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(v, &array);                   CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);                             CHKERRQ(ierr);
    if (i == ndisplaycomp - 1) {ierr = PetscDrawPause(draw); CHKERRQ(ierr);}
    ierr = PetscDrawSave(draw);                              CHKERRQ(ierr);
  }

  ierr = PetscFree(displaycomp); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Mesh(Vec v, PetscViewer viewer){
  DM             dm;
  PetscBool      isdraw;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetDM(v, &dm); CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector not generated from a DM");
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW,  &isdraw); CHKERRQ(ierr);
  if (isdraw) {
    Vec        locv;
    const char *name;

    ierr = DMGetLocalVector(dm, &locv);                      CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) v, &name);       CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) locv, name);     CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv);   CHKERRQ(ierr);
    ierr = VecView_Mesh_Local_Draw(locv, viewer);            CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locv);                  CHKERRQ(ierr);
  } else {
    ierr = VecView_Plex(v, viewer);                          CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



#include <petsc/private/dmimpl.h>

static PetscErrorCode MeshDMPlexView_Draw(DM dm, PetscViewer viewer){
  PetscDraw          draw;
  DM                 cdm;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords;
  PetscReal          bound[4] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscBool          isnull;
  PetscInt           dim, cStart, cEnd, c, N;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;
  const char         *name;
  DMLabel            ghostLabel;

  PetscFunctionBeginUser;
  ierr = PetscViewerDrawGetDraw(viewer, 0, &draw); CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw, &isnull);           CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = DMGetCoordinateDim(dm, &dim); CHKERRQ(ierr);
  if (dim != 2) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes of dimension %D", dim);

  ierr = DMGetCoordinateDM(dm, &cdm);                   CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);         CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);       CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) dm, &name);   CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw, name);                 CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &N);              CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);         CHKERRQ(ierr);
  for (c = 0; c < N; c += dim) {
    bound[0] = PetscMin(bound[0], PetscRealPart(coords[c]));     bound[2] = PetscMax(bound[2], PetscRealPart(coords[c]));
    bound[1] = PetscMin(bound[1], PetscRealPart(coords[c + 1])); bound[3] = PetscMax(bound[3], PetscRealPart(coords[c + 1]));
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);                                                          CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, bound + 0, 2, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, bound + 2, 2, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(draw, bound[0], bound[1], bound[2], bound[3]);                              CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);                                                                               CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);                                            CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);                                                               CHKERRQ(ierr);

  for (c = cStart; c < cEnd; c++) {
    PetscInt ghostVal;
    ierr = DMLabelGetValue(ghostLabel, c, &ghostVal); CHKERRQ(ierr);
    if (ghostVal > 0) continue;

    PetscScalar    *coords = NULL;
    DMPolytopeType ct;
    PetscInt       numCoords;
    ierr = DMPlexGetCellType(dm, c, &ct);                                              CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, &numCoords, &coords); CHKERRQ(ierr);
    switch (ct) {
    case DM_POLYTOPE_TRIANGLE:
      ierr = PetscDrawTriangle(draw,
                               PetscRealPart(coords[0]), PetscRealPart(coords[1]),
                               PetscRealPart(coords[2]), PetscRealPart(coords[3]),
                               PetscRealPart(coords[4]), PetscRealPart(coords[5]),
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      ierr = PetscDrawTriangle(draw,
                               PetscRealPart(coords[0]), PetscRealPart(coords[1]),
                               PetscRealPart(coords[2]), PetscRealPart(coords[3]),
                               PetscRealPart(coords[4]), PetscRealPart(coords[5]),
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2); CHKERRQ(ierr);
      ierr = PetscDrawTriangle(draw,
                               PetscRealPart(coords[0]), PetscRealPart(coords[1]),
                               PetscRealPart(coords[4]), PetscRealPart(coords[5]),
                               PetscRealPart(coords[6]), PetscRealPart(coords[7]),
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      break;
    case DM_POLYTOPE_FV_GHOST:
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of type %s", DMPolytopeTypes[ct]);
      break;
    }
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, &numCoords, &coords); CHKERRQ(ierr);
  }

  ierr = PetscDrawFlush(draw); CHKERRQ(ierr);
  ierr = PetscDrawPause(draw); CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MeshDMView(DM dm, PetscViewer viewer){
  PetscErrorCode ierr;
  PetscBool      isdraw;

  PetscFunctionBeginUser;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW, &isdraw); CHKERRQ(ierr);
  if (isdraw) {
    ierr = MeshDMPlexView_Draw(dm, viewer);                                      CHKERRQ(ierr);
  } else {
    ierr = dm->ops->view(dm, viewer);                                            CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MeshDMSetViewer(DM dm){
  PetscFunctionBeginUser;
  ((PetscObject) dm)->bops->view =  (PetscErrorCode (*)(PetscObject, PetscViewer)) MeshDMView;
  PetscFunctionReturn(0);
}
