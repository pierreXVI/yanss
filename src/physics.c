#include "physics.h"

#define INPUT_FILE "/home/pierre/c/yanss/data/input.yaml"

static const enum ProblemType problem_type = TYPE_EULER;
static struct FieldDescription fields_euler[] = {{"rho", DOF_1},
                                                 {"rho * U", DOF_DIM},
                                                 {"rho * E", DOF_1},
                                                 {NULL, DOF_NULL}};


PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx){
  Physics   phys = (Physics) ctx;

  PetscFunctionBeginUser;
  PrimitiveToConservative(phys, phys->init, u);
  PetscFunctionReturn(0);
}

/*____________________________________________________________________________________________________________________*/

#include "io.h"

void PrimitiveToConservative(Physics phys, const PetscReal in[], PetscReal out[]){
  PetscFunctionBeginUser;
  PetscReal norm2 = 0;
  for (PetscInt i = 0; i < phys->dim; i++){norm2 += PetscSqr(in[1 + i]);}
  out[0] = in[0];
  for (PetscInt i = 0; i < phys->dim; i++){out[1 + i] = in[1 + i] * out[0];}
  out[phys->dof - 1] = in[phys->dof - 1] / (phys->gamma - 1) + 0.5 * norm2 * out[0];
  PetscFunctionReturnVoid();
}
void ConservativeToPrimitive(Physics phys, const PetscReal in[], PetscReal out[]){
  PetscFunctionBeginUser;
  PetscReal norm2 = 0;
  for (PetscInt i = 0; i < phys->dim; i++){norm2 += PetscSqr(in[1 + i]);}

  out[0] = in[0];
  for (PetscInt i = 0; i < phys->dim; i++){out[1 + i] = in[1 + i] / out[0];}
  out[phys->dof - 1] = (phys->gamma - 1) * (in[phys->dof - 1] - 0.5 * norm2 / out[0]);
  PetscFunctionReturnVoid();
}

/*
  Load the BC
  Register the BC in the PetscDS
  Pre-process them if needed
*/
PetscErrorCode PhysSetupBC(Physics phys, DM dm, const char *input_file){
  PetscErrorCode ierr;
  PetscDS system;
  DMLabel label;
  IS is;
  const PetscInt *indices;

  PetscFunctionBeginUser;
  ierr = DMGetLabel(dm, "Face Sets", &label);                                 CHKERRQ(ierr);
  ierr = DMLabelGetNumValues(label, &phys->nbc);                              CHKERRQ(ierr);
  ierr = PetscMalloc1(phys->nbc, &phys->bc);                                  CHKERRQ(ierr);
  ierr = PetscMalloc1(phys->nbc, &phys->bc_ctx);                              CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(label, &is);                                       CHKERRQ(ierr);
  ierr = ISGetIndices(is, &indices);                                          CHKERRQ(ierr);
  ierr = DMGetDS(dm, &system);                                                CHKERRQ(ierr);
  for (PetscInt i; i < phys->nbc; i++) {
    ierr = IOLoadBC(input_file, indices[i], phys->dim, phys->bc + i); CHKERRQ(ierr);
    phys->bc_ctx[i].phys = phys;
    phys->bc_ctx[i].i = i;
    switch (phys->bc[i].type) {
    case BC_DIRICHLET:
      PrimitiveToConservative(phys, phys->bc[i].val, phys->bc[i].val); CHKERRQ(ierr);
      ierr = PetscDSAddBoundary(system, DM_BC_NATURAL_RIEMANN, phys->bc[i].name, "Face Sets", 0, 0, NULL,
                                (void (*)(void)) BCDirichlet, 1, indices + i, phys->bc_ctx + i); CHKERRQ(ierr);
      break;
    case BC_OUTFLOW_P:
      ierr = PetscDSAddBoundary(system, DM_BC_NATURAL_RIEMANN, phys->bc[i].name, "Face Sets", 0, 0, NULL,
                                (void (*)(void)) BCOutflow_P, 1, indices + i, phys->bc_ctx + i); CHKERRQ(ierr);
      break;
    case BC_WALL:
      ierr = PetscDSAddBoundary(system, DM_BC_NATURAL_RIEMANN, phys->bc[i].name, "Face Sets", 0, 0, NULL,
                                (void (*)(void)) BCWall, 1, indices + i, phys->bc_ctx + i);      CHKERRQ(ierr);
      break;
    }
  }

  ierr = ISRestoreIndices(is, &indices);                                      CHKERRQ(ierr);
  ierr = ISDestroy(&is);                                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode PhysicsDestroy(Physics *phys){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < (*phys)->nbc; i++){
    ierr = PetscFree((*phys)->bc[i].name); CHKERRQ(ierr);
    ierr = PetscFree((*phys)->bc[i].val);  CHKERRQ(ierr);
  }
  ierr = PetscFree((*phys)->bc);           CHKERRQ(ierr);
  ierr = PetscFree((*phys)->bc_ctx);       CHKERRQ(ierr);
  ierr = PetscFree((*phys)->init);         CHKERRQ(ierr);
  ierr = PetscFree(*phys);                 CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PhysicsCreate(Physics *phys, DM mesh){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscNew(phys);                      CHKERRQ(ierr);
  ierr = DMGetDimension(mesh, &(*phys)->dim); CHKERRQ(ierr);

  const char *buffer;
  ierr = IOLoadVarFromLoc(INPUT_FILE, "gamma", 0, NULL, &buffer); CHKERRQ(ierr);
  (*phys)->gamma = atof(buffer);
  PetscFree(buffer);

  (*phys)->type = problem_type;
  (*phys)->fields = fields_euler;
  for ((*phys)->nfields = 0, (*phys)->dof = 0; (*phys)->fields[(*phys)->nfields].name; (*phys)->nfields++) {
    switch ((*phys)->fields[(*phys)->nfields].dof) {
    case DOF_1:
      (*phys)->fields[(*phys)->nfields].dof = 1;
      break;
    case DOF_DIM:
      (*phys)->fields[(*phys)->nfields].dof = (*phys)->dim;
      break;
    default: break;
    }
    (*phys)->dof += (*phys)->fields[(*phys)->nfields].dof;
  }

  PetscFV fvm;
  ierr = DMGetField(mesh, 0, NULL, (PetscObject*) &fvm);                                  CHKERRQ(ierr);
  ierr = PetscFVSetSpatialDimension(fvm, (*phys)->dim);                                   CHKERRQ(ierr);
  ierr = PetscFVSetNumComponents(fvm, (*phys)->dof);                                      CHKERRQ(ierr);
  for (PetscInt i = 0, dof = 0; i < (*phys)->nfields; i++){
    if ((*phys)->fields[i].dof == 1) {
      ierr = PetscFVSetComponentName(fvm, dof, (*phys)->fields[i].name);                  CHKERRQ(ierr);
    }
    else {
      for (PetscInt j = 0; j < (*phys)->fields[i].dof; j++){
        char buffer[32];
        ierr = PetscSNPrintf(buffer, sizeof(buffer),"%s_%d", (*phys)->fields[i].name, j); CHKERRQ(ierr);
        ierr = PetscFVSetComponentName(fvm, dof + j, buffer);                             CHKERRQ(ierr);
      }
    }
    dof += (*phys)->fields[i].dof;
  }
  ierr = PetscFVSetFromOptions(fvm);                                                      CHKERRQ(ierr);

  PetscDS system;
  ierr = DMCreateDS(mesh);                                              CHKERRQ(ierr);
  ierr = DMGetDS(mesh, &system);                                        CHKERRQ(ierr);
  ierr = PetscDSSetRiemannSolver(system, 0, RiemannSolver_Euler_Exact); CHKERRQ(ierr);
  ierr = PetscDSSetContext(system, 0, (*phys));                         CHKERRQ(ierr);
  ierr = PhysSetupBC(*phys, mesh, INPUT_FILE);                          CHKERRQ(ierr);
  ierr = PetscDSSetFromOptions(system);                                 CHKERRQ(ierr);

  ierr = IOLoadInitialCondition(INPUT_FILE, (*phys)->dim, &(*phys)->init); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
