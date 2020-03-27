#include "structures.h"
#include "spatial.h"
#include "temporal.h"
#include "physics.h"

static char help[] = "Second Order TVD Finite Volume Example.\n";

int main(int argc, char **argv){
  PetscErrorCode    ierr;

  Physics           phys;
  DM                mesh;
  PetscFV           fvm;

  PetscReal         ftime, dt, minRadius, maxspeed;
  PetscInt          nsteps;
  TS                ts;
  TSConvergedReason reason;
  Vec               x0;


  ierr = PetscOptionsSetValue(NULL, "-draw_pause", "-1");       // TODO: for debugging purposes
  ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return ierr;


  ierr = PhysicsCreate_Advect(&phys);CHKERRQ(ierr);
  /* Count number of fields and dofs */
  for (phys->nfields=0, phys->dof=0; phys->field_desc[phys->nfields].name; phys->nfields++) phys->dof += phys->field_desc[phys->nfields].dof;
  if (phys->dof <= 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Physics did not set dof");

  /* collect max maxspeed from all processes -- todo */
  ierr = MPI_Allreduce(&phys->maxspeed, &maxspeed, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD); CHKERRQ(ierr);
  if (maxspeed <= 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE, "Physics did not set maxspeed");


  /* Create mesh */
  ierr = SetMesh(PETSC_COMM_WORLD, &mesh, &fvm, phys); CHKERRQ(ierr);

  ierr = DMPlexTSGetGeometryFVM(mesh, NULL, NULL, &minRadius);CHKERRQ(ierr);
  //     CFL       * dx        / c;
  dt   = (0.9 * 4) * minRadius / maxspeed;

  ierr = SetTs(PETSC_COMM_WORLD, &ts, mesh, dt); CHKERRQ(ierr);
  ierr = DMTSSetBoundaryLocal(mesh, DMPlexTSComputeBoundary, NULL);CHKERRQ(ierr);
  ierr = DMTSSetRHSFunctionLocal(mesh, DMPlexTSComputeRHSFunctionFVM, NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(mesh, &x0);CHKERRQ(ierr);
  ierr = GetInitialCondition(mesh, x0, phys);CHKERRQ(ierr);
  ierr = TSSolve(ts, x0);CHKERRQ(ierr);

  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&nsteps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,nsteps);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFree(phys);CHKERRQ(ierr);
  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fvm);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
