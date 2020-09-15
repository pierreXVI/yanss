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
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-draw_comp", displaycomp, &ndisplaycomp, &flg); CHKERRQ(ierr);
  if (!ndisplaycomp) ndisplaycomp = Nc;
  *components = displaycomp;
  *outcomponents = ndisplaycomp;
  PetscFunctionReturn(0);
}



/*
  Draws the mesh on the given PetscDraw
*/
static PetscErrorCode PetscDraw_MeshDM_Cells(PetscDraw draw, DM dm){
  PetscErrorCode ierr;
  PetscInt       cStart, cEnd;
  DM             cdm;
  DMLabel        ghostLabel;
  PetscSection   coordSection;
  Vec            coordinates;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateDM(dm, &cdm);                  CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);        CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);      CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL); CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &cEnd, NULL);   CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);         CHKERRQ(ierr);

  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt ghostVal;
    ierr = DMLabelGetValue(ghostLabel, c, &ghostVal); CHKERRQ(ierr);
    if (ghostVal > 0) continue;

    PetscScalar    *coords = NULL;
    DMPolytopeType ct;
    ierr = DMPlexGetCellType(dm, c, &ct);                                        CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, NULL, &coords); CHKERRQ(ierr);
    switch (ct) {
    case DM_POLYTOPE_TRIANGLE:
      ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of type %s", DMPolytopeTypes[ct]);
      break;
    }
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, NULL, &coords); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Draws the border of each partition on the given PetscDraw
*/
static PetscErrorCode PetscDraw_MeshDM_Partition(PetscDraw draw, DM dm){
  PetscErrorCode ierr;
  PetscInt       cStart, cEnd;
  DM             cdm;
  DMLabel        ghostLabel;
  PetscSection   coordSection;
  Vec            coordinates;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinateDM(dm, &cdm);                  CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);        CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);      CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL); CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &cEnd, NULL);   CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);         CHKERRQ(ierr);

  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt ghostVal;
    ierr = DMLabelGetValue(ghostLabel, c, &ghostVal); CHKERRQ(ierr);
    if (ghostVal == -1) continue;

    PetscInt cone_size;
    const PetscInt *cone;

    ierr = DMPlexGetConeSize(dm, c, &cone_size); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &cone);          CHKERRQ(ierr);
    for (PetscInt i = 0; i < cone_size; i++){
      PetscInt ghostVal;
      ierr = DMLabelGetValue(ghostLabel, cone[i], &ghostVal); CHKERRQ(ierr);
      if (ghostVal > 0) continue;

      PetscScalar *coords = NULL;
      ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, cone[i], NULL, &coords); CHKERRQ(ierr);
      ierr = PetscDrawLine(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PETSC_DRAW_BLACK); CHKERRQ(ierr);
      ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, cone[i], NULL, &coords); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}



/*
  Draws a Mesh based vector
  Vector components can be selected with '-draw_comp', and scaling bounds with '-vec_view_bounds'
*/
static PetscErrorCode VecView_Mesh_Local_Draw(Vec v, PetscViewer viewer){
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;

  DM dm;
  {
    PetscBool isnull;
    PetscDraw draw;
    ierr = PetscViewerDrawGetDraw(viewer, 0, &draw); CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw, &isnull);           CHKERRQ(ierr);
    if (isnull) PetscFunctionReturn(0);

    PetscInt  dim;
    ierr = VecGetDM(v, &dm);             CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dm, &dim); CHKERRQ(ierr);
    if (dim != 2) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes of dimension %D.", dim);
  }

  Vec               coordinates;
  PetscSection      coordSection;
  PetscInt          cStart, cEnd, step;
  PetscReal         time;
  DMLabel           ghostLabel;
  const char        *name;
  const PetscScalar *array;
  { // Reading data
    DM cdm;
    ierr = DMGetCoordinateDM(dm, &cdm);                  CHKERRQ(ierr);
    ierr = DMGetLocalSection(cdm, &coordSection);        CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);      CHKERRQ(ierr);

    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL); CHKERRQ(ierr);
    ierr = DMPlexGetGhostCellStratum(dm, &cEnd, NULL);   CHKERRQ(ierr);

    ierr = DMGetLabel(dm, "ghost", &ghostLabel);         CHKERRQ(ierr);

    ierr = PetscObjectGetName((PetscObject) v, &name);   CHKERRQ(ierr);
    ierr = DMGetOutputSequenceNumber(dm, &step, &time);  CHKERRQ(ierr);

    ierr = VecGetArrayRead(v, &array); CHKERRQ(ierr);
  }

  PetscReal bound[4] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  { // Mesh bounds (min for each dimension, max for each dimension)
    PetscInt          size;
    const PetscScalar *coords;

    ierr = VecGetLocalSize(coordinates, &size);   CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords); CHKERRQ(ierr);
    for (PetscInt c = 0; c < size; c += 2) {
      bound[0] = PetscMin(bound[0], coords[c]);     bound[2] = PetscMax(bound[2], coords[c]);
      bound[1] = PetscMin(bound[1], coords[c + 1]); bound[3] = PetscMax(bound[3], coords[c + 1]);
    }
    ierr = VecRestoreArrayRead(coordinates, &coords);                                                          CHKERRQ(ierr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE, bound,     2, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE, bound + 2, 2, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
  }

  PetscInt     ndisplaycomp, *comp, Nc;
  PetscReal    *vbound_tot;
  PetscBool    flg_vbound;
  PetscSection s;
  { // Selecting components
    ierr = DMGetLocalSection(dm, &s);                                               CHKERRQ(ierr);
    ierr = PetscSectionGetFieldComponents(s, 0, &Nc);                               CHKERRQ(ierr);
    ierr = PetscSectionSelectFieldComponents(s, 0, &ndisplaycomp, &comp);    CHKERRQ(ierr);

    const char *prefix;
    PetscInt   nmax = 2 * ndisplaycomp;
    ierr = PetscMalloc1(nmax, &vbound_tot);
    ierr = PetscObjectGetOptionsPrefix((PetscObject) v, &prefix); CHKERRQ(ierr);
    ierr = PetscOptionsGetRealArray(NULL, prefix, "-vec_view_bounds", vbound_tot, &nmax, &flg_vbound); CHKERRQ(ierr);
    if (nmax < 2 * ndisplaycomp) {
      for (PetscInt i = nmax; i < 2 * ndisplaycomp; i++) vbound_tot[i] = (i % 2) ? PETSC_MAX_REAL : PETSC_MIN_REAL;
    }
  }

  for (PetscInt i = 0; i < ndisplaycomp; ++i) {
    PetscDraw  draw, popup;
    const char *cname;
    char       title[PETSC_MAX_PATH_LEN];
    ierr = PetscSectionGetComponentName(s, 0, comp[i], &cname);                                       CHKERRQ(ierr);
    ierr = PetscSNPrintf(title, sizeof(title), "%s:%s Step: %D Time: %.4g", name, cname, step, time); CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer, i, &draw);                                                  CHKERRQ(ierr);
    ierr = PetscDrawSetTitle(draw, title);                                                            CHKERRQ(ierr);

    PetscReal vbound[2] = {0, 0};
    if (flg_vbound) {vbound[0] = vbound_tot[2 * i]; vbound[1] = vbound_tot[2 * i + 1];}
    if (vbound[0] >= vbound[1]) {
      Vec subv;
      IS  is;
      ierr = ISCreateStride(PetscObjectComm((PetscObject) v), cEnd - cStart, comp[i], Nc, &is); CHKERRQ(ierr);
      ierr = VecGetSubVector(v, is, &subv);                                                     CHKERRQ(ierr);
      ierr = VecMin(subv, NULL, &vbound[0]);                                                    CHKERRQ(ierr);
      ierr = VecMax(subv, NULL, &vbound[1]);                                                    CHKERRQ(ierr);
      ierr = VecRestoreSubVector(v, is, &subv);                                                 CHKERRQ(ierr);
      ierr = ISDestroy(&is);                                                                    CHKERRQ(ierr);
      if (vbound[1] <= vbound[0]) vbound[1] = vbound[0] + 1.0;
      ierr = MPIU_Allreduce(MPI_IN_PLACE, vbound + 0, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
      ierr = MPIU_Allreduce(MPI_IN_PLACE, vbound + 1, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
    }
    ierr = PetscDrawGetPopup(draw, &popup);                                       CHKERRQ(ierr);
    ierr = PetscDrawScalePopup(popup, vbound[0], vbound[1]);                      CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates(draw, bound[0], bound[1], bound[2], bound[3]); CHKERRQ(ierr);

    for (PetscInt c = cStart; c < cEnd; c++) {
      PetscInt ghostVal;
      ierr = DMLabelGetValue(ghostLabel, c, &ghostVal); CHKERRQ(ierr);
      if (ghostVal > 0) continue;

      PetscScalar    *coords = NULL, *a = NULL;
      PetscInt       color[4];
      DMPolytopeType ct;

      ierr = DMPlexPointLocalRead(dm, c, array, &a); CHKERRQ(ierr);
      if (a) {
        color[1] = color[2] = color[3] = color[0] = PetscDrawRealToColor(PetscRealPart(a[comp[i]]), vbound[0], vbound[1]);
      } else {
        PetscScalar *vals = NULL;
        PetscInt     numVals, va;

        ierr = DMPlexVecGetClosure(dm, NULL, v, c, &numVals, &vals); CHKERRQ(ierr);
        if (numVals % Nc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The number of components %D does not divide the number of values in the closure %D", Nc, numVals);
        switch (numVals / Nc) {
        case 3: /* P1 Triangle */
        case 4: /* P1 Quadrangle */
          for (va = 0; va < numVals/Nc; ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va * Nc + comp[i]]), vbound[0], vbound[1]);
          break;
        case 6: /* P2 Triangle */
        case 8: /* P2 Quadrangle */
          for (va = 0; va < numVals / (2*Nc); ++va) color[va] = PetscDrawRealToColor(PetscRealPart(vals[va * Nc + comp[i] + numVals / (2 * Nc)]), vbound[0], vbound[1]);
          break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of values for cell closure %D cannot be handled", numVals/Nc);
        }
        ierr = DMPlexVecRestoreClosure(dm, NULL, v, c, &numVals, &vals); CHKERRQ(ierr);
      }

      ierr = DMPlexGetCellType(dm, c, &ct);                                        CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, NULL, &coords); CHKERRQ(ierr);
      switch (ct) {
      case DM_POLYTOPE_TRIANGLE:
        ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]); CHKERRQ(ierr);
        break;
      case DM_POLYTOPE_QUADRILATERAL:
        ierr = PetscDrawTriangle(draw, PetscRealPart(coords[0]), PetscRealPart(coords[1]), PetscRealPart(coords[2]), PetscRealPart(coords[3]), PetscRealPart(coords[4]), PetscRealPart(coords[5]), color[0], color[1], color[2]); CHKERRQ(ierr);
        ierr = PetscDrawTriangle(draw, PetscRealPart(coords[4]), PetscRealPart(coords[5]), PetscRealPart(coords[6]), PetscRealPart(coords[7]), PetscRealPart(coords[0]), PetscRealPart(coords[1]), color[2], color[3], color[0]); CHKERRQ(ierr);
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of type %s", DMPolytopeTypes[ct]);
        break;
      }
      ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, NULL, &coords); CHKERRQ(ierr);
    }

    // ierr = PetscDraw_MeshDM_Cells(draw, dm); CHKERRQ(ierr);
    ierr = PetscDraw_MeshDM_Partition(draw, dm); CHKERRQ(ierr);

    ierr = PetscDrawFlush(draw);                             CHKERRQ(ierr);
    if (i == ndisplaycomp - 1) {ierr = PetscDrawPause(draw); CHKERRQ(ierr);}
    ierr = PetscDrawSave(draw);                              CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(v, &array);                   CHKERRQ(ierr);
  ierr = PetscFree(vbound_tot); CHKERRQ(ierr);
  ierr = PetscFree(comp);       CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Mesh(Vec v, PetscViewer viewer){
  DM             dm;
  PetscBool      isdraw;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetDM(v, &dm);                                                        CHKERRQ(ierr);
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
    PETSC_EXTERN PetscErrorCode VecView_Plex(Vec, PetscViewer);
    ierr = VecView_Plex(v, viewer); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



/*
  Draws a 2D Mesh
*/
static PetscErrorCode MeshDMPlexView_Draw(DM dm, PetscViewer viewer){
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;

  PetscDraw draw;
  {
    PetscBool isnull;
    ierr = PetscViewerDrawGetDraw(viewer, 0, &draw); CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw, &isnull);           CHKERRQ(ierr);
    if (isnull) PetscFunctionReturn(0);

    PetscInt  dim;
    ierr = DMGetCoordinateDim(dm, &dim); CHKERRQ(ierr);
    if (dim != 2) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot draw meshes of dimension %D.", dim);
  }

  Vec               coordinates;
  PetscSection      coordSection;
  PetscInt          cStart, cEnd;
  DMLabel           ghostLabel;
  PetscMPIInt       rank;

  { // Reading data
    DM cdm;
    ierr = DMGetCoordinateDM(dm, &cdm);                             CHKERRQ(ierr);
    ierr = DMGetLocalSection(cdm, &coordSection);                   CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);                 CHKERRQ(ierr);

    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);            CHKERRQ(ierr);
    ierr = DMPlexGetGhostCellStratum(dm, &cEnd, NULL);              CHKERRQ(ierr);

    ierr = DMGetLabel(dm, "ghost", &ghostLabel);                    CHKERRQ(ierr);

    const char *name;
    ierr = PetscObjectGetName((PetscObject) dm, &name);             CHKERRQ(ierr);
    ierr = PetscDrawSetTitle(draw, name);                           CHKERRQ(ierr);
    ierr = PetscDrawClear(draw);                                    CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank); CHKERRQ(ierr);
  }


  { // Mesh bounds (min for each dimension, max for each dimension)
    PetscReal         bound[4] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
    PetscInt          size;
    const PetscScalar *coords;

    ierr = VecGetLocalSize(coordinates, &size);   CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords); CHKERRQ(ierr);
    for (PetscInt c = 0; c < size; c += 2) {
      bound[0] = PetscMin(bound[0], coords[c]);     bound[2] = PetscMax(bound[2], coords[c]);
      bound[1] = PetscMin(bound[1], coords[c + 1]); bound[3] = PetscMax(bound[3], coords[c + 1]);
    }
    ierr = VecRestoreArrayRead(coordinates, &coords);                                                          CHKERRQ(ierr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE, bound,     2, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE, bound + 2, 2, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) dm)); CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates(draw, bound[0], bound[1], bound[2], bound[3]);                              CHKERRQ(ierr);
  }


  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt ghostVal;
    ierr = DMLabelGetValue(ghostLabel, c, &ghostVal); CHKERRQ(ierr);
    if (ghostVal > 0) continue;

    PetscScalar    *coords = NULL;
    DMPolytopeType ct;
    ierr = DMPlexGetCellType(dm, c, &ct);                                        CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, NULL, &coords); CHKERRQ(ierr);
    switch (ct) {
    case DM_POLYTOPE_TRIANGLE:
      ierr = PetscDrawTriangle(draw,
                               PetscRealPart(coords[0]), PetscRealPart(coords[1]),
                               PetscRealPart(coords[2]), PetscRealPart(coords[3]),
                               PetscRealPart(coords[4]), PetscRealPart(coords[5]),
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2,
                               PETSC_DRAW_WHITE + rank % (PETSC_DRAW_BASIC_COLORS - 2) + 2); CHKERRQ(ierr);
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
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot draw cells of type %s", DMPolytopeTypes[ct]);
      break;
    }
    ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, NULL, &coords); CHKERRQ(ierr);
  }

  ierr = PetscDraw_MeshDM_Cells(draw, dm); CHKERRQ(ierr);

  ierr = PetscDrawFlush(draw); CHKERRQ(ierr);
  ierr = PetscDrawPause(draw); CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#include <petsc/private/dmimpl.h>

/*
  View a Mesh
  Calls MeshDMPlexView_Draw if the viewer is of type PETSCVIEWERDRAW, else calls the native viewer (DMView_Plex)
*/
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