#include "physics.h"


PetscErrorCode BCDirichlet(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscReal *xI, PetscReal *xG, void *ctx){
  struct BCCtx *bc_ctx = (struct BCCtx*) ctx;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < bc_ctx->phys->dof; i++) xG[i] = bc_ctx->phys->bc[bc_ctx->i].val[i];
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
  PetscReal alpha = bc_ctx->phys->bc[bc_ctx->i].val[0] / wI[bc_ctx->phys->dim + 1] - 1; // p / pi - 1

  PetscReal wG[bc_ctx->phys->dof];
  wG[0] = (1 + alpha / bc_ctx->phys->gamma) * wI[0];
  for (PetscInt i = 0; i < bc_ctx->phys->dim; i++) wG[1 + i] = wG[0] * (wI[1 + i] - (alpha / bc_ctx->phys->gamma) * ci * n[i] / area); // u <- u - (alpha/gamma)*c n
  wG[bc_ctx->phys->dim + 1] = bc_ctx->phys->bc[bc_ctx->i].val[0];
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
