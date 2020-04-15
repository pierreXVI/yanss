#include "physics.h"
#include "spatial.h"
#include "temporal.h"

static char help[] = "Second Order TVD Finite Volume Example.\n";
static char mesh_filename[] = "/home/pierre/c/yanss/data/box.msh";  // TODO


int main(int argc, char **argv){
  PetscErrorCode    ierr;

  DM                mesh;
  Physics           phys;
  TS                ts;
  Vec               x0;

  // TODO: for debugging purposes
  PetscOptionsSetValue(NULL, "-options_left", "no");
  PetscOptionsSetValue(NULL, "-draw_pause", "-1");
  // PetscOptionsSetValue(NULL, "-vec_view_bounds", "0,10,0,10,0,10,2E5,5E5");

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;

  ierr = MeshLoad(PETSC_COMM_WORLD, mesh_filename, &mesh); CHKERRQ(ierr);
  ierr = PhysicsCreate(&phys, mesh);                       CHKERRQ(ierr);
  ierr = MeshCreateGlobalVector(mesh, &x0);                CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x0, "State");    CHKERRQ(ierr);

  // TODO
  PetscReal dt, minRadius;
  ierr = DMPlexTSGetGeometryFVM(mesh, NULL, NULL, &minRadius); CHKERRQ(ierr);
  dt   = (0.1) * minRadius / (PetscMax(U_0, U_1) + PetscSqrtReal(phys->gamma * P_0 / RHO_0));

  ierr = MyTsCreate(PETSC_COMM_WORLD, &ts, mesh, phys, dt);      CHKERRQ(ierr);
  ierr = MeshApplyFunction(mesh, 0, InitialCondition, phys, x0); CHKERRQ(ierr);
  ierr = TSSolve(ts, x0);                                        CHKERRQ(ierr);

  PetscReal         ftime;
  PetscInt          nsteps;
  TSConvergedReason reason;
  ierr = TSGetSolveTime(ts, &ftime);        CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &nsteps);      CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts, &reason); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %D steps\n", TSConvergedReasons[reason], (double)ftime, nsteps); CHKERRQ(ierr);

  ierr = VecDestroy(&x0);       CHKERRQ(ierr);
  ierr = TSDestroy(&ts);        CHKERRQ(ierr);
  ierr = PhysicsDestroy(&phys); CHKERRQ(ierr);
  ierr = MeshDestroy(&mesh);    CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
