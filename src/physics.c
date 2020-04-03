#include "physics.h"

static const enum ProblemType problem_type = TYPE_ADVECTION;

// static const PetscReal c_advec[] = {1, 0, 0};

// static struct FieldDescription fields_advec[] = {{"U", DOF_DIM}, {NULL, DOF_NULL}};
static struct FieldDescription fields_euler[] = {{"rho", DOF_1},
                                                 {"rho * U", DOF_DIM},
                                                 {"rho * E", DOF_1},
                                                 {NULL, DOF_NULL}};

static PetscReal bc_inflow[] = {1, 1, 0, 2.5E5};
static struct BCDescription bc[] = {{"wall", BC_WALL, NULL},
                                    {"outflow", BC_OUTFLOW, NULL},
                                    {"inflow", BC_DIRICHLET, bc_inflow},
                                    {NULL, BC_NULL, NULL}};

static PetscReal initial_condition[] = {1, 0, 0, 2.5E5};
PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx){
  PetscInt i;

  PetscFunctionBeginUser;
  for (i = 0; i < Nf; i++){
    u[i] = initial_condition[i];
  }
  PetscFunctionReturn(0);
}

/*____________________________________________________________________________________________________________________*/

const char * const BCTypes[] = {"BC_NULL", "Dirichlet", "Outflow", "Wall"};

/*
static void RiemannSolver_Advec(PetscInt dim, PetscInt Nf,
                                const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[],
                                PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx){
  PetscReal dot = 0;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < dim; i++){
    dot += c_advec[i] * n[i];
  }
  for (PetscInt i = 0; i < Nf; i++){
    flux[i] = (dot > 0 ? uL[i] : uR[i]) * dot;
  }
  PetscFunctionReturnVoid();
}
*/

static void RiemannSolver_Euler_Centred(PetscInt dim, PetscInt Nf,
                                const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[],
                                PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx){
  Physics phys = (Physics) ctx;
  PetscReal dot = 0, square = 0, uI[Nf], pI;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < Nf; i++){
    uI[i] = (uL[i] + uR[i]) / 2;
  }
  for (PetscInt i = 0; i < dim; i++){ // dot = U.n, square = U^2
    dot +=  n[i] * uI[1 + i] / uI[0];
    square += uI[1 + i] * uI[1 + i];
  }
  pI = (phys->gamma - 1) * (uI[Nf - 1] - square/(2 * uI[0]));
  // PetscPrintf(PETSC_COMM_WORLD, "In %f, %f, pressure is %f bar\n", x[0], x[1], pI/1E5);
  // PetscPrintf(PETSC_COMM_WORLD, "In %f, %f, %f, %f\n", (phys->gamma - 1), uI[Nf - 1], square, uI[0]);
  // CHKERRABORT(PETSC_COMM_WORLD, 1);

  for (PetscInt i = 0; i < Nf; i++){ // (U.n) * x
    flux[i] = dot * uI[i];
  }
  for (PetscInt i = 0; i < dim; i++){ // + p * n_i for rho * U_i
    flux[i] +=  pI * n[i];
  }
  flux[Nf - 1] += dot * pI; // + (U.n) * p for rho * E

  PetscFunctionReturnVoid();
}


static PetscErrorCode BCDirichlet(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscScalar *xI, PetscScalar *xG, void *ctx){
  struct BC_ctx *bc_ctx = (struct BC_ctx*) ctx;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < bc_ctx->phys->dof; i++){
    xG[i] = bc_ctx->phys->bc[bc_ctx->i].val[i];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode BCOutflow(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscScalar *xI, PetscScalar *xG, void *ctx){
  struct BC_ctx *bc_ctx = (struct BC_ctx*) ctx;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < bc_ctx->phys->dof; i++){
    xG[i] = xI[i];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode BCWall(PetscReal time, const PetscReal c[3], const PetscReal n[3], const PetscScalar *xI, PetscScalar *xG, void *ctx){
  struct BC_ctx *bc_ctx = (struct BC_ctx*) ctx;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < bc_ctx->phys->dof; i++){
    xG[i] = xI[i];
  }

  switch (bc_ctx->phys->type) {
  case TYPE_NS: /* u <- 0 */
    for (PetscInt i = 0; i < bc_ctx->phys->dim; i++){
      xG[1 + i] = 0;
    }
    break;
  case TYPE_EULER: /* u <- u - (u.n)n */
    ;
    PetscReal dot = 0;
    for (PetscInt i = 0; i < bc_ctx->phys->dim; i++){
      dot += xI[1 + i] * n[i];
    }
    for (PetscInt i = 0; i < bc_ctx->phys->dim; i++){
      xG[1 + i] = xI[1 + i] - dot * n[i];
    }
    break;
  default: /* is an outflow */
    break;
  }
  PetscFunctionReturn(0);
}


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

  (*phys)->bc = bc;
  (*phys)->nbc = 0; while ((*phys)->bc[(*phys)->nbc].name) {(*phys)->nbc++;}
  ierr = PetscMalloc1((*phys)->nbc, &((*phys)->bc_ctx)); CHKERRQ(ierr);

  PetscDS system;
  ierr = DMCreateDS(mesh);                                                                                CHKERRQ(ierr);
  ierr = DMGetDS(mesh, &system);                                                                          CHKERRQ(ierr);
  ierr = PetscDSSetRiemannSolver(system, 0, RiemannSolver_Euler_Centred);                                 CHKERRQ(ierr);
  ierr = PetscDSSetContext(system, 0, (*phys));                                                           CHKERRQ(ierr);
  for (PetscInt i = 1; i <= (*phys)->nbc; i++) {
    (*phys)->bc_ctx[i - 1].phys = *phys;
    (*phys)->bc_ctx[i - 1].i = i - 1;

    switch ((*phys)->bc[i - 1].type) {
      case BC_DIRICHLET:
        ierr = PetscDSAddBoundary(system, DM_BC_ESSENTIAL, (*phys)->bc[i - 1].name, "Face Sets", 0, 0, NULL,
                                  (void (*)(void)) BCDirichlet, 1, &i, &(*phys)->bc_ctx[i - 1]);          CHKERRQ(ierr);
        break;
      case BC_OUTFLOW:
        ierr = PetscDSAddBoundary(system, DM_BC_NATURAL, (*phys)->bc[i - 1].name, "Face Sets", 0, 0, NULL,
                                  (void (*)(void)) BCOutflow, 1, &i, &(*phys)->bc_ctx[i - 1]);            CHKERRQ(ierr);
        break;
      case BC_WALL:
        ierr = PetscDSAddBoundary(system, DM_BC_NATURAL, (*phys)->bc[i - 1].name, "Face Sets", 0, 0, NULL,
                                  (void (*)(void)) BCWall, 1, &i, &(*phys)->bc_ctx[i - 1]);               CHKERRQ(ierr);
        break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unknown boundary condition: %s\n", BCTypes[(*phys)->bc[i - 1].type]);
        break;
    }
  }
  ierr = PetscDSSetFromOptions(system);                                                                   CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
