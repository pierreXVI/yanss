#include "physics.h"

static const enum ProblemType problem_type = TYPE_EULER;
static struct FieldDescription fields_euler[] = {{"rho", DOF_1},
                                                 {"rho * U", DOF_DIM},
                                                 {"rho * E", DOF_1},
                                                 {NULL, DOF_NULL}};

#define RHO_0 1
#define U_0   0
#define U_1   0
#define P_0   1E5

PetscReal bc_inflow[4] = {RHO_0, U_1, 0, P_0};
PetscReal bc_outflow[1] = {P_0};
static struct BCDescription bc[] = {{"wall", BC_WALL, NULL},
                                    {"outflow", BC_OUTFLOW_P, bc_outflow},
                                    {"inflow", BC_DIRICHLET, bc_inflow},
                                    {NULL, BC_NULL, NULL}};

PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx){
  Physics   phys = (Physics) ctx;

  PetscFunctionBeginUser;
  u[0] = RHO_0;
  for (PetscInt i = 0; i < dim; i++){u[1 + i] = 0;}
  u[1] = RHO_0 * U_0;
  u[Nf - 1] = P_0 / (phys->gamma - 1) + 0.5 * RHO_0 * PetscSqr(U_0);
  PetscFunctionReturn(0);
}

/*____________________________________________________________________________________________________________________*/

const char * const ProblemTypes[] = {"Euler", "Navier-Stokes"};

PetscErrorCode PhysicsDestroy(Physics *phys){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree((*phys)->bc_ctx); CHKERRQ(ierr);
  ierr = PetscFree(*phys);           CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PhysicsCreate(Physics *phys, DM mesh){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscNew(phys);                      CHKERRQ(ierr);
  ierr = DMGetDimension(mesh, &(*phys)->dim); CHKERRQ(ierr);
  (*phys)->type = problem_type;
  (*phys)->gamma = 1.4;
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
  ierr = DMGetField(mesh, 0, NULL, (PetscObject*) &fvm);                               CHKERRQ(ierr);
  ierr = PetscFVSetSpatialDimension(fvm, (*phys)->dim);                                CHKERRQ(ierr);
  ierr = PetscFVSetNumComponents(fvm, (*phys)->dof);                                   CHKERRQ(ierr);
  for (PetscInt i = 0, dof = 0; i < (*phys)->nfields; i++){
    if ((*phys)->fields[i].dof == 1) {
      ierr = PetscFVSetComponentName(fvm, dof, (*phys)->fields[i].name);               CHKERRQ(ierr);
    }
    else {
      for (PetscInt j = 0; j < (*phys)->fields[i].dof; j++){
        static PetscInt buffer_size = 32;
        char buffer[buffer_size];
        ierr = PetscSNPrintf(buffer, buffer_size,"%s_%d", (*phys)->fields[i].name, j); CHKERRQ(ierr);
        ierr = PetscFVSetComponentName(fvm, dof + j, buffer);                          CHKERRQ(ierr);
      }
    }
    dof += (*phys)->fields[i].dof;
  }
  ierr = PetscFVSetFromOptions(fvm);                                                   CHKERRQ(ierr);


  PetscDS system;
  ierr = DMCreateDS(mesh);                                                                                CHKERRQ(ierr);
  ierr = DMGetDS(mesh, &system);                                                                          CHKERRQ(ierr);
  ierr = PetscDSSetRiemannSolver(system, 0, RiemannSolver_Euler_Exact);                                   CHKERRQ(ierr);
  ierr = PetscDSSetContext(system, 0, (*phys));                                                           CHKERRQ(ierr);
  ierr = PhysSetupBC(*phys, system, bc);                         CHKERRQ(ierr);

  ierr = PetscDSSetFromOptions(system);                                                                   CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PhysSetupBC(Physics phys, PetscDS system, struct BCDescription *bc){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  phys->bc = bc;
  while (bc[phys->nbc].name) {phys->nbc++;}
  ierr = PetscMalloc1(phys->nbc, &phys->bc_ctx); CHKERRQ(ierr);

  for (PetscInt i = 1; i <= phys->nbc; i++) {
    phys->bc_ctx[i - 1].phys = phys;
    phys->bc_ctx[i - 1].i = i - 1;

    switch (phys->bc[i - 1].type) {
    case BC_DIRICHLET:
      ierr = PetscDSAddBoundary(system, DM_BC_NATURAL_RIEMANN, phys->bc[i - 1].name, "Face Sets", 0, 0, NULL,
                                (void (*)(void)) BCDirichlet, 1, &i, &phys->bc_ctx[i - 1]); CHKERRQ(ierr);

      // Convert from primitive (r, u, p) to conservative (r, ru, rE)
      PetscReal norm2 = 0;
      for (PetscInt j = 0; j < phys->dim; j++) {
        norm2 += PetscSqr(bc[i - 1].val[1 + j]);
        bc[i - 1].val[1 + j] *= bc[i - 1].val[0];
      }
      bc[i - 1].val[phys->dof - 1] = bc[i - 1].val[phys->dof - 1] / (phys->gamma - 1) + 0.5 * bc[i - 1].val[0] * norm2;
      break;
    case BC_OUTFLOW_P:
      ierr = PetscDSAddBoundary(system, DM_BC_NATURAL_RIEMANN, phys->bc[i - 1].name, "Face Sets", 0, 0, NULL,
                                (void (*)(void)) BCOutflow_P, 1, &i, &phys->bc_ctx[i - 1]); CHKERRQ(ierr);
      break;
    case BC_WALL:
      ierr = PetscDSAddBoundary(system, DM_BC_NATURAL_RIEMANN, phys->bc[i - 1].name, "Face Sets", 0, 0, NULL,
                                (void (*)(void)) BCWall, 1, &i, &phys->bc_ctx[i - 1]);      CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unknown boundary condition (%d)\n", phys->bc[i - 1].type);
      break;
    }
  }
  PetscFunctionReturn(0);
}
