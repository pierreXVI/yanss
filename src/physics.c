#include "physics.h"

static const enum ProblemType problem_type = TYPE_EULER;

// static const PetscReal c_advec[] = {1, 0, 0};

// static struct FieldDescription fields_advec[] = {{"U", DOF_DIM}, {NULL, DOF_NULL}};
static struct FieldDescription fields_euler[] = {{"rho", DOF_1},
                                                 {"rho * U", DOF_DIM},
                                                 {"rho * E", DOF_1},
                                                 {NULL, DOF_NULL}};

static PetscReal bc_inflow[4] = {1, 1 * 0.1, 0, 2.5E5 + 0.5 * 1 * PetscSqr(0.1)};
static struct BCDescription bc[] = {{"wall", BC_WALL, NULL},
                                    {"outflow", BC_OUTFLOW, NULL},
                                    {"inflow", BC_DIRICHLET, bc_inflow},
                                    {NULL, BC_NULL, NULL}};

static const PetscReal r0 = 1;
static const PetscReal u0[] = {0, 0, 0};
static const PetscReal p0 = 1E5;
PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx){
  Physics   phys = (Physics) ctx;
  PetscReal norm2 = 0;

  PetscFunctionBeginUser;
  u[0] = r0;
  for (PetscInt i = 0; i < dim; i++){
    u[1 + i] = r0 * u0[i];
    norm2 += PetscSqr(u0[i]);
  }
  u[Nf - 1] = p0 / (phys->gamma - 1) + 0.5 * r0 * norm2;
  PetscFunctionReturn(0);
}


/*____________________________________________________________________________________________________________________*/

const char * const BCTypes[] = {"BC_NULL", "Dirichlet", "Outflow", "Wall"};
const char * const ProblemTypes[] = {"Advection", "Euler", "Navier-Stokes"};

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

static void RiemannSolver_Euler_Exact(PetscInt dim, PetscInt Nf,
                                const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[],
                                PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx){
  Physics phys = (Physics) ctx;

  PetscFunctionBeginUser;

  PetscReal area = 0, nn[dim]; // n = area * nn, nn normal vector to the surface
  for (PetscInt i = 0; i < dim; i++){area += PetscSqr(n[i]);}
  area = PetscSqrtReal(area);
  for (PetscInt i = 0; i < dim; i++){nn[i] = n[i] / area;}


  PetscReal rl = uL[0], rr = uR[0]; // left and right densities
  PetscReal ul = 0, ur = 0; // left and right normal speeds
  for (PetscInt i = 0; i < dim; i++){
    ul += uL[1 + i] * nn[i];
    ur += uR[1 + i] * nn[i];
  }
  ul /= rl;
  ur /= rr;
  PetscReal utl[dim], utr[dim]; // left and right tangent speeds
  for (PetscInt i = 0; i < dim; i++){
    utl[i] = uL[1 + i] / rl - ul * nn[i];
    utr[i] = uR[1 + i] / rr - ur * nn[i];
  }
  PetscReal norm2l = 0, norm2r = 0;
  for (PetscInt i = 0; i < dim; i++){
    norm2l += PetscSqr(uL[1 + i]);
    norm2r += PetscSqr(uR[1 + i]);
  }
  PetscReal pl = (phys->gamma - 1) * (uL[Nf - 1] - 0.5 * norm2l / rl); // left pressure
  PetscReal pr = (phys->gamma - 1) * (uR[Nf - 1] - 0.5 * norm2r / rr); // right pressure

  PetscReal cl = PetscSqrtReal(phys->gamma * pl / rl); // left speed of sound
  PetscReal cr = PetscSqrtReal(phys->gamma * pr / rr); // right speed of sound

  PetscReal alpha = (phys->gamma + 1) / (2 * phys->gamma);
  PetscReal beta  = (phys->gamma - 1) / (2 * phys->gamma);
  PetscReal delta = (phys->gamma + 1) / (phys->gamma - 1);

  PetscReal pstar = PetscPowReal((beta * (ul - ur) + cl + cr) / (cl * PetscPowReal(pl, -beta) + cr * PetscPowReal(pr, -beta)), 1 / beta);
  PetscReal ml, mr;
  PetscBool flg_sl, flg_sr;
  for (PetscInt i = 0; i < N_ITER_RIEMANN; i++) {
    PetscReal pratiol = pstar / pl;
    PetscReal pratior = pstar / pr;
    flg_sl = pratiol >= 1;
    flg_sr = pratior >= 1;
    ml = (flg_sl) ? rl * cl * PetscSqrtReal(1 + alpha * (pratiol - 1)) : rl * cl * beta * (1 - pratiol) / (1 - PetscPowReal(pratiol, beta));
    mr = (flg_sr) ? rr * cr * PetscSqrtReal(1 + alpha * (pratior - 1)) : rr * cr * beta * (1 - pratior) / (1 - PetscPowReal(pratior, beta));
    pstar = (mr * pl + ml * pr + ml * mr * (ul - ur)) / (ml + mr);
    if (pstar < 0) {SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "ERROR : p* < 0");}
  }

  PetscReal ustar = (ml * ul + mr * ur + pl - pr) / (ml + mr);

  PetscReal rout, uout, *utout, pout; // density, normal and tangent speeds, and pressure at the interface
  if (ustar > 0) {
    utout = utl;
    PetscReal pratiol = pstar / pl;
    if (flg_sl) { // shock -> uL*
      uout = ustar;
      pout = pstar;
      rout = rl * (1 + delta * pratiol) / (delta + pratiol);
    } else {
      PetscReal rstarl = rl * PetscPowReal(pratiol, 1 / phys->gamma);
      PetscReal cstarl = PetscSqrtReal(phys->gamma * pstar / rstarl);
      if (ustar < cstarl) { // rarefaction -> uL*
        uout = ustar;
        pout = pstar;
        rout = rstarl;
      } else { // sonic rarefaction
        uout = (ul * cstarl - ustar * cl) / (ul - ustar + cstarl - cl);
        pout = pl * PetscPowReal(uout / cl, 1 / beta);
        rout = rl * PetscPowReal(pout / pl, 1 / phys->gamma);
      }
    }
  } else {
    utout = utr;
    PetscReal pratior = pstar / pr;
    if (flg_sr) { // shock -> uR*
      uout = ustar;
      pout = pstar;
      rout = rr * (1 + delta * pratior) / (delta + pratior);
    } else {
      PetscReal rstarr = rr * PetscPowReal(pratior, 1 / phys->gamma);
      PetscReal cstarr = PetscSqrtReal(phys->gamma * pstar / rstarr);
      if (ustar < cstarr) { // rarefaction -> uR*
        uout = ustar;
        pout = pstar;
        rout = rstarr;
      } else { // sonic rarefaction
        uout = (ur * cstarr - ustar * cr) / (ur - ustar + cstarr - cr);
        pout = pr * PetscPowReal(uout / cr, 1 / beta);
        rout = rr * PetscPowReal(pout / pr, 1 / phys->gamma);
      }
    }
  }

  PetscReal un = uout * area, unorm2 = PetscSqr(uout);
  for (PetscInt i = 0; i < dim; i++) {unorm2 += PetscSqr(utout[i]);}

  flux[0] = rout * un;
  for (PetscInt i = 0; i < dim; i++) {flux[1 + i] = rout * (uout * nn[i] + utout[i]) * un + pout * n[i];}
  flux[Nf - 1] = (pout * phys->gamma / (phys->gamma - 1) + 0.5 * rout * unorm2) * un;

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
  ierr = PetscDSSetRiemannSolver(system, 0, RiemannSolver_Euler_Exact);                                   CHKERRQ(ierr);
  ierr = PetscDSSetContext(system, 0, (*phys));                                                           CHKERRQ(ierr);
  for (PetscInt i = 1; i <= (*phys)->nbc; i++) {
    (*phys)->bc_ctx[i - 1].phys = *phys;
    (*phys)->bc_ctx[i - 1].i = i - 1;

    switch ((*phys)->bc[i - 1].type) {
      case BC_DIRICHLET:
        ierr = PetscDSAddBoundary(system, DM_BC_NATURAL_RIEMANN, (*phys)->bc[i - 1].name, "Face Sets", 0, 0, NULL,
                                  (void (*)(void)) BCDirichlet, 1, &i, &(*phys)->bc_ctx[i - 1]);          CHKERRQ(ierr);
        break;
      case BC_OUTFLOW:
        ierr = PetscDSAddBoundary(system, DM_BC_NATURAL_RIEMANN, (*phys)->bc[i - 1].name, "Face Sets", 0, 0, NULL,
                                  (void (*)(void)) BCOutflow, 1, &i, &(*phys)->bc_ctx[i - 1]);            CHKERRQ(ierr);
        break;
      case BC_WALL:
        ierr = PetscDSAddBoundary(system, DM_BC_NATURAL_RIEMANN, (*phys)->bc[i - 1].name, "Face Sets", 0, 0, NULL,
                                  (void (*)(void)) BCWall, 1, &i, &(*phys)->bc_ctx[i - 1]);               CHKERRQ(ierr);
        break;
      default:
        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unknown boundary condition (%d)\n", (*phys)->bc[i - 1].type);
        break;
    }
  }
  ierr = PetscDSSetFromOptions(system);                                                                   CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
