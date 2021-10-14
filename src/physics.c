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
                    {NULL, 0}};


PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nc, PetscReal *u, void *ctx){
  Physics phys = (Physics) ctx;
  PetscFunctionBeginUser;

  PetscReal M = 0.5, beta = 0.2, R = 0.005;
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
  // PrimitiveToConservative(phys, phys->init, u);

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


void normU(const PetscReal *x, PetscReal *y, void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;
  *y = 0;
  for (PetscInt i = 0; i < phys->dim; i++) *y += PetscSqr(x[1 + i] / x[0]);
  *y = PetscSqrtReal(*y);
  PetscFunctionReturnVoid();
}

void mach(const PetscReal *x, PetscReal *y, void *ctx){
  Physics   phys = (Physics) ctx;
  PetscReal w[phys->dof];

  PetscFunctionBeginUser;
  *y = 0;
  ConservativeToPrimitive(phys, x, w);
  for (PetscInt i = 0; i < phys->dim; i++) *y += PetscSqr(w[1 + i]);
  *y = PetscSqrtReal(*y * w[0] / (phys->gamma * w[phys->dim + 1]));
  PetscFunctionReturnVoid();
}



PetscErrorCode PhysicsDestroy(Physics *phys){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < (*phys)->nbc; i++) {
    ierr = PetscFree((*phys)->bc_ctx[i].type); CHKERRQ(ierr);
    ierr = PetscFree((*phys)->bc_ctx[i].name); CHKERRQ(ierr);
    ierr = PetscFree((*phys)->bc_ctx[i].val);  CHKERRQ(ierr);
  }
  ierr = PetscFree((*phys)->bc_ctx);           CHKERRQ(ierr);
  ierr = PetscFree((*phys)->init);             CHKERRQ(ierr);
  ierr = PetscFree(*phys);                     CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PhysicsCreate(Physics *phys, const char *filename, DM dm){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscNew(phys);                    CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &(*phys)->dim); CHKERRQ(ierr);

  { // Read values from input file
    const char *buffer, *loc = "Physics";
    ierr = YAMLLoadVarFromLoc(filename, "gamma", 1, &loc, &buffer); CHKERRQ(ierr);
    (*phys)->gamma = atof(buffer);
    ierr = PetscFree(buffer); CHKERRQ(ierr);

    ierr = YAMLLoadInitialCondition(filename, (*phys)->dim, &(*phys)->init); CHKERRQ(ierr);
  }

  { // Setting fields
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
    char buffer[64];
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);                           CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvm, (*phys)->dof);                             CHKERRQ(ierr);
    for (PetscInt i = 0, dof = 0; i < nfields; i++) {
      if (fields[i].dof == 1) {
        ierr = PetscFVSetComponentName(fvm, dof, fields[i].name);                  CHKERRQ(ierr);
      }
      else {
        for (PetscInt j = 0; j < fields[i].dof; j++) {
          ierr = PetscSNPrintf(buffer, sizeof(buffer),"%s_%d", fields[i].name, j); CHKERRQ(ierr);
          ierr = PetscFVSetComponentName(fvm, dof + j, buffer);                    CHKERRQ(ierr);
        }
      }
      dof += fields[i].dof;
    }
  }

  ierr = PhysicsRiemannSetFromOptions(PetscObjectComm((PetscObject) dm), *phys); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
