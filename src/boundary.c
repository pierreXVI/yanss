#include "physics.h"

/*____________________________________________________________________________________________________________________*/

const char * const BCTypes[] = {"BC_NULL", "Dirichlet", "Outflow_P", "Wall"};

PetscErrorCode BCDirichlet(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscScalar *xI, PetscScalar *xG, void *ctx){
  struct BC_ctx *bc_ctx = (struct BC_ctx*) ctx;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < bc_ctx->phys->dof; i++){
    xG[i] = bc_ctx->phys->bc[bc_ctx->i].val[i];
  }
  PetscPrintf(PETSC_COMM_WORLD, "BCDirichlet : xI = %3f, % 3f, % 3f, %.3E, xG = %3f, % 3f, % 3f, %.3E\n", xI[0], xI[1], xI[2], xI[3], xG[0], xG[1], xG[2], xG[3]);
  PetscFunctionReturn(0);
}

PetscErrorCode BCOutflow_P(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscScalar *xI, PetscScalar *xG, void *ctx){
  struct BC_ctx *bc_ctx = (struct BC_ctx*) ctx;

  PetscFunctionBeginUser;
  PetscReal norm2 = 0, area = 0;
  for (PetscInt i = 0; i < bc_ctx->phys->dim; i++) {
    norm2 += PetscSqr(xI[1 + i]);
    area += PetscSqr(n[i]);
  }
  area = PetscSqrtReal(area);

  PetscReal pi = (bc_ctx->phys->gamma - 1) * (xI[bc_ctx->phys->dof - 1] - 0.5 * norm2 / xI[0]);
  PetscReal alpha = bc_ctx->phys->bc[bc_ctx->i].val[0]/pi - 1;
  PetscReal ci = PetscSqrtReal(bc_ctx->phys->gamma * pi / xI[0]);

  norm2 = 0;
  xG[0] = (1 + alpha / bc_ctx->phys->gamma) * xI[0];
  for (PetscInt i = 0; i < bc_ctx->phys->dim; i++){
    xG[1 + i] = xG[0] * (xI[1 + i] - alpha * ci * n[i] / (area * bc_ctx->phys->gamma));
    norm2 += PetscSqr(xG[1 + i]);
  }
  xG[bc_ctx->phys->dof - 1] = bc_ctx->phys->bc[bc_ctx->i].val[0] / (bc_ctx->phys->gamma - 1) + 0.5 * norm2 / xG[0];
  PetscPrintf(PETSC_COMM_WORLD, "BCOutflow_P : xI = %3f, % 3f, % 3f, %.3E, xG = %3f, % 3f, % 3f, %.3E\n", xI[0], xI[1], xI[2], xI[3], xG[0], xG[1], xG[2], xG[3]);

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
  default: /* TODO */
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Wall boundary condition not implemented for this physics (%d)\n", bc_ctx->phys->type);
    break;
  }
  PetscPrintf(PETSC_COMM_WORLD, "BCWall      : xI = %3f, % 3f, % 3f, %.3E, xG = %3f, % 3f, % 3f, %.3E\n", xI[0], xI[1], xI[2], xI[3], xG[0], xG[1], xG[2], xG[3]);
  PetscFunctionReturn(0);
}
