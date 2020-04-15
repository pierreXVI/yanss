#include "physics.h"

#define MONITOR_BC


const char * const BCTypes[] = {"BC_NULL", "Dirichlet", "Outflow_P", "Wall"};

PetscErrorCode BCDirichlet(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscScalar *xI, PetscScalar *xG, void *ctx){
  struct BC_ctx *bc_ctx = (struct BC_ctx*) ctx;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < bc_ctx->phys->dof; i++){
    xG[i] = bc_ctx->phys->bc[bc_ctx->i].val[i];
  }
#ifdef MONITOR_BC
  PetscPrintf(PETSC_COMM_WORLD, "BCDirichlet : xI = %3f, % 3f, % 3f, %.3E, xG = %3f, % 3f, % 3f, %.3E\n", xI[0], xI[1], xI[2], xI[3], xG[0], xG[1], xG[2], xG[3]);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode BCOutflow_P(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscScalar *xI, PetscScalar *xG, void *ctx){
  struct BC_ctx *bc_ctx = (struct BC_ctx*) ctx;

  PetscFunctionBeginUser;
  PetscReal area = 0;
  for (PetscInt i = 0; i < bc_ctx->phys->dim; i++) {area += PetscSqr(n[i]);}
  area = PetscSqrtReal(area);

  PetscScalar wI[bc_ctx->phys->dof];
  ConservativeToPrimitive(bc_ctx->phys, xI, wI);
  PetscReal ci = PetscSqrtReal(bc_ctx->phys->gamma * wI[bc_ctx->phys->dof - 1] / wI[0]);
  PetscReal alpha = bc_ctx->phys->bc[bc_ctx->i].val[0] / wI[bc_ctx->phys->dof - 1] - 1; // p / pi - 1

  PetscScalar wG[bc_ctx->phys->dof];
  wG[0] = (1 + alpha / bc_ctx->phys->gamma) * wI[0];
  for (PetscInt i = 0; i < bc_ctx->phys->dim; i++){
    wG[1 + i] = wG[0] * (wI[1 + i] - (alpha / bc_ctx->phys->gamma) * ci * n[i] / area); // u <- u - (alpha/gamma)*c n
  }
  wG[bc_ctx->phys->dof - 1] = bc_ctx->phys->bc[bc_ctx->i].val[0];
  PrimitiveToConservative(bc_ctx->phys, wG, xG);
#ifdef MONITOR_BC
  PetscPrintf(PETSC_COMM_WORLD, "BCOutflow_P : xI = %3f, % 3f, % 3f, %.3E, xG = %3f, % 3f, % 3f, %.3E\n", xI[0], xI[1], xI[2], xI[3], xG[0], xG[1], xG[2], xG[3]);
#endif

  PetscFunctionReturn(0);
}

PetscErrorCode BCWall(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscScalar *xI, PetscScalar *xG, void *ctx){
  struct BC_ctx *bc_ctx = (struct BC_ctx*) ctx;

  PetscFunctionBeginUser;
  switch (bc_ctx->phys->type) {
  case TYPE_EULER: /* u <- u - (u.n)n */
    xG[0] = xI[0];
    xG[bc_ctx->phys->dof - 1] = xI[bc_ctx->phys->dof - 1];

    PetscReal dot = 0, norm2 = 0;
    for (PetscInt i = 0; i < bc_ctx->phys->dim; i++){
      dot += xI[1 + i] * n[i];
      norm2 += PetscSqr(n[i]);
    }
    for (PetscInt i = 0; i < bc_ctx->phys->dim; i++){xG[1 + i] = xI[1 + i] - 2 * dot * n[i] / norm2;}
    break;
  case TYPE_NS: /* u <- 0 */
    xG[0] = xI[0];
    xG[bc_ctx->phys->dof - 1] = xI[bc_ctx->phys->dof - 1];
    for (PetscInt i = 0; i < bc_ctx->phys->dim; i++){xG[1 + i] = -xI[1 + i];}
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Wall boundary condition not implemented for this physics (%d)\n", bc_ctx->phys->type);
    break;
  }
#ifdef MONITOR_BC
  PetscPrintf(PETSC_COMM_WORLD, "BCWall      : xI = %3f, % 3f, % 3f, %.3E, xG = %3f, % 3f, % 3f, %.3E\n", xI[0], xI[1], xI[2], xI[3], xG[0], xG[1], xG[2], xG[3]);
#endif
  PetscFunctionReturn(0);
}
