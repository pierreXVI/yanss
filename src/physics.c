#include "physics.h"
#include "input.h"
#include "spatial.h"


/*____________________________________________________________________________________________________________________*/
static struct FieldDescription {
  const char *name_c;
  const char *name_p;
  PetscInt   dof;
} FIELDS_NS[] = {
  {"rho",     "rho", DOF_1},
  {"rho * U", "U",   DOF_DIM},
  {"rho * E", "p",   DOF_1},
  {NULL,      NULL,  0}
};


PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nc, PetscReal *u, void *ctx){
  Physics phys = (Physics) ctx;
  PetscFunctionBeginUser;

  PetscReal M = 0.8, beta = 1.3860642817598832, R = 1;
  PetscReal Xc = 0, Yc = 0;
  PetscReal Pinf = 1, Tinf = 1;
  PetscReal rhoinf = Pinf / (phys->r_gas * Tinf);

  PetscReal Uinf = M * PetscSqrtReal(phys->gamma * phys->r_gas * Tinf);

  PetscReal dx = (x[0] - Xc) / R, dy = (x[1] - Yc) / R;
  PetscReal alpha = beta * PetscExpReal(-(PetscSqr(dx) + PetscSqr(dy)) / 2);

  PetscReal T0 = Tinf - PetscSqr(Uinf * alpha) * (phys->gamma - 1) / (2 * phys->gamma * phys->r_gas);
  PetscReal rho0 = rhoinf * PetscPowReal(T0 / Tinf, 1 / (phys->gamma - 1));

  u[0] = rho0;
  u[1] = Uinf * (1 - alpha * dy);
  u[2] = Uinf * alpha * dx;
  u[3] = rho0 * phys->r_gas * T0;

  PrimitiveToConservative(u, u, phys);
  // PrimitiveToConservative(phys->init, u, phys);

  PetscFunctionReturn(0);
}
/*____________________________________________________________________________________________________________________*/



void PrimitiveToConservative(const PetscReal in[], PetscReal out[], Physics phys){
  PetscFunctionBeginUser;
  PetscReal norm2 = 0;
  for (PetscInt i = 0; i < phys->dim; i++) norm2 += PetscSqr(in[1 + i]);

  out[0] = in[0];
  for (PetscInt i = 0; i < phys->dim; i++) out[1 + i] = in[1 + i] * out[0];
  out[phys->dim + 1] = in[phys->dim + 1] / (phys->gamma - 1) + 0.5 * norm2 * out[0];
  PetscFunctionReturnVoid();
}

void ConservativeToPrimitive(const PetscReal in[], PetscReal out[], Physics phys){
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
  ConservativeToPrimitive(x, w, phys);
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
    ierr = YAMLLoadVarFromLoc(filename, "r_gas", 1, &loc, &buffer); CHKERRQ(ierr);
    (*phys)->r_gas = atof(buffer);
    ierr = YAMLLoadVarFromLoc(filename, "mu", 1, &loc, &buffer); CHKERRQ(ierr);
    (*phys)->mu = atof(buffer);
    ierr = YAMLLoadVarFromLoc(filename, "lambda", 1, &loc, &buffer); CHKERRQ(ierr);
    (*phys)->lambda = atof(buffer);
    ierr = PetscFree(buffer); CHKERRQ(ierr);

    ierr = YAMLLoadInitialCondition(filename, (*phys)->dim, &(*phys)->init); CHKERRQ(ierr);
  }

  { // Setting fields
    struct FieldDescription *fields = FIELDS_NS;
    PetscInt nfields;
    for (nfields = 0, (*phys)->dof = 0; fields[nfields].name_c; nfields++) {
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

    DM      dmGrad;
    char    buffer[64];
    PetscFV fvm, fvmGrad;
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);                                       CHKERRQ(ierr);
    ierr = DMPlexGetDataFVM(dm, fvm, NULL, NULL, &dmGrad);                                     CHKERRQ(ierr);
    ierr = DMGetField(dmGrad, 0, NULL, (PetscObject*) &fvmGrad);                               CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvm, (*phys)->dof);                                         CHKERRQ(ierr);
    ierr = PetscFVSetNumComponents(fvmGrad, (*phys)->dim * (*phys)->dof);                      CHKERRQ(ierr);
    for (PetscInt i = 0, dof = 0; i < nfields; i++) {
      if (fields[i].dof == 1) {
        ierr = PetscFVSetComponentName(fvm, dof, fields[i].name_c);                            CHKERRQ(ierr);
        for (PetscInt k = 0; k < (*phys)->dim; k++) {
          ierr = PetscSNPrintf(buffer, sizeof(buffer),"d_%d %s", k, fields[i].name_p);         CHKERRQ(ierr);
          ierr = PetscFVSetComponentName(fvmGrad, (*phys)->dim * dof + k, buffer);             CHKERRQ(ierr);
        }
      } else {
        for (PetscInt j = 0; j < fields[i].dof; j++) {
          ierr = PetscSNPrintf(buffer, sizeof(buffer),"%s_%d", fields[i].name_c, j);           CHKERRQ(ierr);
          ierr = PetscFVSetComponentName(fvm, dof + j, buffer);                                CHKERRQ(ierr);
          for (PetscInt k = 0; k < (*phys)->dim; k++) {
            ierr = PetscSNPrintf(buffer, sizeof(buffer),"d_%d %s_%d", k, fields[i].name_p, j); CHKERRQ(ierr);
            ierr = PetscFVSetComponentName(fvmGrad, (*phys)->dim * (dof + j) + k, buffer);     CHKERRQ(ierr);
          }
        }
      }
      dof += fields[i].dof;
    }
  }

  ierr = PhysicsRiemannSetFromOptions(PetscObjectComm((PetscObject) dm), *phys); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
