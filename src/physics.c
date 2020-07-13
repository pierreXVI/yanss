#include "physics.h"
#include "input.h"
#include "spatial.h"


/*____________________________________________________________________________________________________________________*/
static const enum ProblemType problem_type = TYPE_EULER;
static struct FieldDescription {
  const char *name;
  PetscInt   dof;
} fields_euler[] = {{"rho", DOF_1},
                    {"rho * U", DOF_DIM},
                    {"rho * E", DOF_1},
                    {PETSC_NULL, 0}};


PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nc, PetscReal *u, void *ctx){
  Physics phys = (Physics) ctx;
  PetscFunctionBeginUser;

  PetscReal M = 0.5, beta = 0.2, R = 0.01;
  PetscReal Xc = 0.05, Yc = 0.05;
  PetscReal Pinf = 1.0e+05, Tinf = 300, Rgas = 287.15;

  PetscReal Uinf = M * PetscSqrtReal(phys->gamma * Rgas * Tinf);

  PetscReal dx = (x[0] - Xc) / R, dy = (x[1] - Yc) / R;
  PetscReal r2 =  PetscSqr(dx) + PetscSqr(dy);
  PetscReal e = PetscExpReal(-r2 / 2);

  PetscReal T0 = Tinf - PetscSqr(Uinf * beta * e) * (phys->gamma - 1) / (2 * phys->gamma * Rgas);
  PetscReal rhoinf = Pinf / (Rgas * Tinf);
  PetscReal rho0 = rhoinf * PetscPowReal(T0 / Tinf, 1 / (phys->gamma - 1));

  u[0] = rho0;
  u[1] = Uinf * (1 - beta * e * dy);
  u[2] = Uinf * beta * e * dx;
  u[3] = Rgas * rho0 * T0;

  PrimitiveToConservative(phys, u, u);

  PetscFunctionReturn(0);
}
/*____________________________________________________________________________________________________________________*/



void PrimitiveToConservative(Physics phys, const PetscReal in[], PetscReal out[]){
  PetscFunctionBeginUser;
  PetscReal norm2 = 0;
  for (PetscInt i = 0; i < phys->dim; i++) norm2 += PetscSqr(in[1 + i]);

  out[0] = in[0];
  for (PetscInt i = 0; i < phys->dim; i++) out[1 + i] = in[1 + i] * out[0];
  out[phys->dim + 1] = in[phys->dim + 1] / (phys->gamma - 1) + 0.5 * norm2 * out[0];
  PetscFunctionReturnVoid();
}

void ConservativeToPrimitive(Physics phys, const PetscReal in[], PetscReal out[]){
  PetscFunctionBeginUser;
  PetscReal norm2 = 0;
  for (PetscInt i = 0; i < phys->dim; i++)norm2 += PetscSqr(in[1 + i]);

  out[0] = in[0];
  for (PetscInt i = 0; i < phys->dim; i++)out[1 + i] = in[1 + i] / out[0];
  out[phys->dim + 1] = (phys->gamma - 1) * (in[phys->dim + 1] - 0.5 * norm2 / out[0]);
  PetscFunctionReturnVoid();
}


PetscErrorCode normU(PetscInt Nc, const PetscReal *x, PetscScalar *y, void *ctx){
  Physics   phys = (Physics) ctx;

  PetscFunctionBeginUser;
  *y = 0;
  for (PetscInt i = 0; i < phys->dim; i++) *y += PetscSqr(x[1 + i] / x[0]);
  *y = PetscSqrtReal(*y);
  PetscFunctionReturn(0);
}

PetscErrorCode mach(PetscInt Nc, const PetscReal *x, PetscScalar *y, void *ctx){
  Physics   phys = (Physics) ctx;
  PetscReal w[phys->dof];

  PetscFunctionBeginUser;
  *y = 0;
  ConservativeToPrimitive(phys, x, w);
  for (PetscInt i = 0; i < phys->dim; i++) *y += PetscSqr(w[1 + i]);
  *y = PetscSqrtReal(*y * w[0] / (phys->gamma * w[phys->dim + 1]));
  PetscFunctionReturn(0);
}



PetscErrorCode PhysicsDestroy(Physics *phys){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecDestroy(&(*phys)->x);          CHKERRQ(ierr);
  for (PetscInt i = 0; i < (*phys)->nbc; i++){
    ierr = PetscFree((*phys)->bc_ctx[i].name); CHKERRQ(ierr);
    ierr = PetscFree((*phys)->bc_ctx[i].val);  CHKERRQ(ierr);
  }
  ierr = PetscFree((*phys)->bc_ctx);           CHKERRQ(ierr);
  ierr = PetscFree((*phys)->init);             CHKERRQ(ierr);
  ierr = PetscFree(*phys);                     CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PhysicsCreate(Physics *phys, const char *filename, Mesh mesh){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscNew(phys);                          CHKERRQ(ierr);
  ierr = DMGetDimension(mesh->dm, &(*phys)->dim); CHKERRQ(ierr);

  const char *buffer, *loc = "Physics";
  ierr = IOLoadVarFromLoc(filename, "gamma", 1, &loc, &buffer); CHKERRQ(ierr);
  (*phys)->gamma = atof(buffer);
  ierr = PetscFree(buffer);                                     CHKERRQ(ierr);

  (*phys)->type = problem_type;
  struct FieldDescription *fields = fields_euler;
  PetscInt nfields;
  for (nfields = 0, (*phys)->dof = 0; fields[nfields].name; nfields++) {
    switch (fields[nfields].dof) {
    case DOF_1:
      fields[nfields].dof = 1;
      break;
    case DOF_DIM:
      fields[nfields].dof = (*phys)->dim;
      break;
    default: break;
    }
    (*phys)->dof += fields[nfields].dof;
  }

  PetscFV fvm;
  ierr = DMGetField(mesh->dm, 0, PETSC_NULL, (PetscObject*) &fvm);               CHKERRQ(ierr);
  ierr = PetscFVSetSpatialDimension(fvm, (*phys)->dim);                          CHKERRQ(ierr);
  ierr = PetscFVSetNumComponents(fvm, (*phys)->dof);                             CHKERRQ(ierr);
  for (PetscInt i = 0, dof = 0; i < nfields; i++){
    if (fields[i].dof == 1) {
      ierr = PetscFVSetComponentName(fvm, dof, fields[i].name);                  CHKERRQ(ierr);
    }
    else {
      for (PetscInt j = 0; j < fields[i].dof; j++){
        char buffer[32];
        ierr = PetscSNPrintf(buffer, sizeof(buffer),"%s_%d", fields[i].name, j); CHKERRQ(ierr);
        ierr = PetscFVSetComponentName(fvm, dof + j, buffer);                    CHKERRQ(ierr);
      }
    }
    dof += fields[i].dof;
  }
  ierr = PetscFVSetFromOptions(fvm);                                             CHKERRQ(ierr);

  PetscDS prob;
  DMLabel label;
  IS      is;
  const PetscInt *indices;
  ierr = DMCreateDS(mesh->dm);                                           CHKERRQ(ierr);
  ierr = DMGetDS(mesh->dm, &prob);                                       CHKERRQ(ierr);
  ierr = PetscDSSetRiemannSolver(prob, 0, RiemannSolver_AdvectionX);     CHKERRQ(ierr);
  ierr = PetscDSSetContext(prob, 0, (*phys));                            CHKERRQ(ierr);
  ierr = DMGetLabel(mesh->dm, "Face Sets", &label);                      CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(label, &(*phys)->nbc);                      CHKERRQ(ierr);
  ierr = PetscMalloc1((*phys)->nbc, &(*phys)->bc_ctx);                   CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(label, &is);                                  CHKERRQ(ierr);
  ierr = ISGetIndices(is, &indices);                                     CHKERRQ(ierr);
  ierr = DMGetDS(mesh->dm, &prob);                                       CHKERRQ(ierr);

  for (PetscInt i = 0; i < (*phys)->nbc; i++) {
    (*phys)->bc_ctx[i].phys = *phys;
    ierr = IOLoadBC(filename, indices[i], (*phys)->dim, (*phys)->bc_ctx + i); CHKERRQ(ierr);

    void (*bcFunc)(void);
    switch ((*phys)->bc_ctx[i].type) {
    case BC_DIRICHLET:
      PrimitiveToConservative(*phys, (*phys)->bc_ctx[i].val, (*phys)->bc_ctx[i].val);
      bcFunc = (void (*)(void)) BCDirichlet;
      break;
    case BC_OUTFLOW_P:
      bcFunc = (void (*)(void)) BCOutflow_P;
      break;
    case BC_WALL:
      bcFunc = (void (*)(void)) BCWall;
      break;
    }
    ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, (*phys)->bc_ctx[i].name, "Face Sets", 0, 0,
                              PETSC_NULL, bcFunc, 1, indices + i, (*phys)->bc_ctx + i); CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(is, &indices);                                 CHKERRQ(ierr);
  ierr = ISDestroy(&is);                                                 CHKERRQ(ierr);
  ierr = PetscDSSetFromOptions(prob);                                    CHKERRQ(ierr);

  ierr = DMTSSetRHSFunctionLocal(mesh->dm, MeshDMTSComputeRHSFunctionFVM, mesh); CHKERRQ(ierr);

  ierr = IOLoadInitialCondition(filename, (*phys)->dim, &(*phys)->init); CHKERRQ(ierr);

  ierr = MeshDMCreateGlobalVector(mesh->dm, &(*phys)->x);                       CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) (*phys)->x, "Solution");      CHKERRQ(ierr);
  ierr = MeshDMApplyFunction(mesh->dm, 0, InitialCondition, *phys, (*phys)->x); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
