#include "physics.h"


PetscErrorCode BCDirichlet(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscReal *xI, PetscReal *xG, void *ctx){
  struct BCCtx *bc_ctx = (struct BCCtx*) ctx;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < bc_ctx->phys->dof; i++) xG[i] = bc_ctx->val[i];
  PetscFunctionReturn(0);
}

PetscErrorCode BCOutflow_P(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscReal *xI, PetscReal *xG, void *ctx){
  struct BCCtx *bc_ctx = (struct BCCtx*) ctx;

  PetscFunctionBeginUser;
  PetscReal area = 0;
  for (PetscInt i = 0; i < bc_ctx->phys->dim; i++) area += PetscSqr(n[i]);
  area = PetscSqrtReal(area);

  PetscReal wI[bc_ctx->phys->dof];
  ConservativeToPrimitive(bc_ctx->phys, xI, wI);
  PetscReal ci = PetscSqrtReal(bc_ctx->phys->gamma * wI[bc_ctx->phys->dim + 1] / wI[0]);
  PetscReal alpha = bc_ctx->val[0] / wI[bc_ctx->phys->dim + 1] - 1; // p / pi - 1

  PetscReal wG[bc_ctx->phys->dof];
  wG[0] = (1 + alpha / bc_ctx->phys->gamma) * wI[0];
  for (PetscInt i = 0; i < bc_ctx->phys->dim; i++) wG[1 + i] = wG[0] * (wI[1 + i] - (alpha / bc_ctx->phys->gamma) * ci * n[i] / area); // u <- u - (alpha/gamma)*c n
  wG[bc_ctx->phys->dim + 1] = bc_ctx->val[0];
  PrimitiveToConservative(bc_ctx->phys, wG, xG);

  PetscFunctionReturn(0);
}

PetscErrorCode BCWall(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscReal *xI, PetscReal *xG, void *ctx){
  struct BCCtx *bc_ctx = (struct BCCtx*) ctx;

  PetscFunctionBeginUser;
  switch (bc_ctx->phys->type) {
  case TYPE_EULER: /* u <- u - (u.n)n */
    xG[0] = xI[0];
    xG[bc_ctx->phys->dim + 1] = xI[bc_ctx->phys->dim + 1];

    PetscReal dot = 0, norm2 = 0;
    for (PetscInt i = 0; i < bc_ctx->phys->dim; i++){
      dot += xI[1 + i] * n[i];
      norm2 += PetscSqr(n[i]);
    }
    for (PetscInt i = 0; i < bc_ctx->phys->dim; i++) xG[1 + i] = xI[1 + i] - 2 * dot * n[i] / norm2;
    break;

  case TYPE_NS: /* u <- 0 */{}
    xG[0] = xI[0];
    xG[bc_ctx->phys->dim + 1] = xI[bc_ctx->phys->dim + 1];

    for (PetscInt i = 0; i < bc_ctx->phys->dim; i++) xG[1 + i] = -xI[1 + i];
    break;

  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Wall boundary condition not implemented for this model (%d)\n", bc_ctx->phys->type);
    break;
  }
  PetscFunctionReturn(0);
}


#define ALMOST_EPS(a, b, c) PetscAbs(PetscAbs((a) - (b)) - (c)) < 1E-5
PetscErrorCode BCPerio(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscReal *xI, PetscReal *xG, void *ctx){
  // struct BCCtx   *bc_ctx = (struct BCCtx*) ctx;
  // PetscErrorCode ierr;
  // DM dm;

  PetscFunctionBeginUser;
  // for (PetscInt i = 0; i < bc_ctx->phys->dof; i++) xG[i] = 0;
  //
  // // PetscPrintf(PETSC_COMM_WORLD, "In        %.2f, %.2f (% .2f, % .2f)\n", c[0], c[1], n[0], n[1]);
  //
  // ierr = VecGetDM(bc_ctx->phys->x, &dm); CHKERRQ(ierr);
  //
  //
  // IS ids;
  // PetscInt size;
  // const PetscInt *values;
  // PetscReal centroid[3], normal[3];
  //
  // // ierr = DMGetLabelIdIS(dm, "Face Sets", &ids); CHKERRQ(ierr);
  // ierr = DMGetStratumIS(dm, "Face Sets", 20, &ids); CHKERRQ(ierr); // IS of all faces in boundary 20
  // ISGetSize(ids, &size);
  // ISGetIndices(ids, &values);
  // PetscBool found = PETSC_FALSE;
  // for (PetscInt i = 0; i < size; i++) {
  //   DMPlexComputeCellGeometryFVM(dm, values[i], PETSC_NULL, centroid, PETSC_NULL); CHKERRQ(ierr);
  //
  //   if ((centroid[0]-c[0])*n[1] - (centroid[1]-c[1])*n[0]) {
  //     found = PETSC_TRUE;
  //   }
  // }
  //
  // if (!found) PetscPrintf(PETSC_COMM_WORLD, "NOT FOUND\n");
  //
  //
  // PetscInt fStart, fEnd, fMe=-1, fOther=-1;
  // ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);
  // for (PetscInt f = fStart; f < fEnd; f++) {
  //   DMPlexComputeCellGeometryFVM(dm, f, PETSC_NULL, centroid, normal); CHKERRQ(ierr);
  //   if (centroid[0] == c[0] && centroid[1] == c[1]) {
  //     fMe = f;
  //   }
  //   if ((ALMOST_EPS(centroid[0], c[0], 0.1) && ALMOST_EPS(centroid[1], c[1], 0)) || (ALMOST_EPS(centroid[1], c[1], 0.1) && ALMOST_EPS(centroid[0], c[0], 0))) { // TODO
  //     fOther = f;
  //     // break;
  //   }
  // }
  // // PetscPrintf(PETSC_COMM_WORLD, "    I am %d <- %d\n", fMe, fOther);
  //
  // PetscInt p1, p2;
  // ierr = DMPlexGetGhostCellStratum(dm, &p1, &p2);  CHKERRQ(ierr);
  //
  // PetscInt nC;
  // PetscInt const *support;
  // DMPlexGetSupportSize(dm, fOther, &nC);
  // DMPlexGetSupport(dm, fOther, &support);
  // // PetscPrintf(PETSC_COMM_WORLD, "    Support is of size %d\n", nC, support[0], support[1], p1, p2);
  //
  // const PetscReal *ar;
  // PetscReal *xSym;
  // ierr = VecGetArrayRead(bc_ctx->phys->x, &ar); CHKERRQ(ierr);
  //
  // for (PetscInt i = 0; i < nC; i++) {
  //   DMPlexComputeCellGeometryFVM(dm, support[i], PETSC_NULL, centroid, normal); CHKERRQ(ierr);
  //   if (support[i] < p2 && support[i] >= p1) {
  //     // PetscPrintf(PETSC_COMM_WORLD, "    Cell %d is ghost\n", support[i]);
  //   } else {
  //     // PetscPrintf(PETSC_COMM_WORLD, "    Cell %d in %f, %f\n", support[i], centroid[0], centroid[1]);
  //     ierr = DMPlexPointLocalRead(dm, support[i], ar, &xSym);CHKERRQ(ierr);
  //     for (PetscInt i = 0; i < bc_ctx->phys->dof; i++) xG[i] = xSym[i];
  //     break;
  //   }
  //
  // ierr = VecRestoreArrayRead(bc_ctx->phys->x, &ar); CHKERRQ(ierr);
  //
  // }

  PetscFunctionReturn(0);
}
