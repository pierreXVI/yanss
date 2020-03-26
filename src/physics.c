#include "physics.h"

PETSC_STATIC_INLINE PetscReal Dot2Real(const PetscReal *x,const PetscReal *y) { return x[0]*y[0] + x[1]*y[1];}
PETSC_STATIC_INLINE PetscReal Norm2Real(const PetscReal *x) { return PetscSqrtReal(PetscAbsReal(Dot2Real(x,x)));}

PetscErrorCode SetBC(PetscDS prob, Physics phys){
  PetscErrorCode ierr;
  const PetscInt inflowids[] = {100,200,300};
  const PetscInt outflowids[] = {101};

  PetscFunctionBeginUser;
  ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "inflow",  "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Advect_Inflow,  3,  inflowids,  phys);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "outflow", "Face Sets", 0, 0, NULL, (void (*)(void)) PhysicsBoundary_Advect_Outflow, 1, outflowids, phys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode PhysicsBoundary_Advect_Inflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  Physics        phys    = (Physics)ctx;

  PetscFunctionBeginUser;
  xG[0] = phys->inflowState;
  PetscFunctionReturn(0);
}
PetscErrorCode PhysicsBoundary_Advect_Outflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  PetscFunctionBeginUser;
  xG[0] = xI[0];
  PetscFunctionReturn(0);
}


PetscErrorCode PhysicsCreate_Advect(Physics phys)
{
  PetscFunctionBeginUser;
  phys->field_desc  = PhysicsFields_Advect;
  phys->riemann     = (PetscRiemannFunc)PhysicsRiemann_Advect;
  phys->wind[0]     = 0.0;
  phys->wind[1]     = 1.0;
  phys->inflowState = -2.0;
  phys->maxspeed    = PetscSqrtReal(PetscAbsReal(phys->wind[0]*phys->wind[0] + phys->wind[1]*phys->wind[1]));
  PetscFunctionReturn(0);
}


PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx){
  Physics        phys    = (Physics)ctx;

  PetscFunctionBeginUser;
  PetscReal             x0[dim];

  x0[0] = x[0] - time * phys->wind[0];
  x0[1] = x[1] - time * phys->wind[1];

  if (x0[1] > 0) u[0] = 1. * x[0] + 3. * x[1];
  else u[0] = phys->inflowState;
  PetscFunctionReturn(0);
}

void PhysicsRiemann_Advect(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, Physics phys)
{
  PetscReal wn;
  wn = phys->wind[0] * n[0] + phys->wind[1] * n[1];
  flux[0] = (wn > 0 ? xL[0] : xR[0]) * wn;
}
