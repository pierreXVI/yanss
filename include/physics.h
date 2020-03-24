#ifndef PHYSICS
#define PHYSICS

#include "structures.h"

PetscErrorCode PhysicsBoundary_Advect_Inflow (PetscReal, const PetscReal*, const PetscReal*, const PetscScalar*, PetscScalar*, void*);
PetscErrorCode PhysicsBoundary_Advect_Outflow(PetscReal, const PetscReal*, const PetscReal*, const PetscScalar*, PetscScalar*, void*);

PetscErrorCode SetBC(PetscDS, Physics);


PetscErrorCode PhysicsCreate_Advect(Physics);

static const struct FieldDescription PhysicsFields_Advect[] = {{"U",1},{NULL,0}};

PetscErrorCode InitialCondition(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*);

void PhysicsRiemann_Advect(PetscInt, PetscInt, const PetscReal*, const PetscReal*, const PetscScalar*, const PetscScalar*, PetscInt, const PetscScalar[], PetscScalar*, Physics);

#endif
