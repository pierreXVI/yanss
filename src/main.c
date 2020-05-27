#include "physics.h"
#include "spatial.h"
#include "temporal.h"
#include "input.h"

static const char help[] = "Finite Volume solver\n";
static const char input_filename[] = "/home/pierre/c/yanss/data/input.yaml";


int main(int argc, char **argv){
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help); if (ierr) return ierr;
  ierr = IOLoadPetscOptions(input_filename); CHKERRQ(ierr);

  PetscBool set;
  char      mesh_filename[256];
  ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-mesh", mesh_filename, sizeof(mesh_filename), &set); CHKERRQ(ierr);
  if (!set) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "No mesh file was given: please use the option \"-mesh filename\"");

  DM      mesh;
  Physics phys;
  ierr = MeshLoadFromFile(PETSC_COMM_WORLD, mesh_filename, &mesh); CHKERRQ(ierr);
  ierr = PhysicsCreate(&phys, input_filename, mesh);               CHKERRQ(ierr);

  // TODO
  PetscReal dt, minRadius=1;
  // ierr = DMPlexTSGetGeometryFVM(mesh, PETSC_NULL, PETSC_NULL, &minRadius); CHKERRQ(ierr);
  ierr = DMPlexGetMinRadius(mesh, &minRadius); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "minRadius = %e\n", minRadius);
  dt   = (0.1) * minRadius / (PetscMax(phys->init[1], phys->init[2]) + PetscSqrtReal(phys->gamma * phys->init[3] / phys->init[0]));
  // dt   = (0.1) * minRadius / (PetscMax(PetscMax(phys->init[1], phys->init[2]), phys->init[3]) + PetscSqrtReal(phys->gamma * phys->init[4] / phys->init[0]));
  PetscPrintf(PETSC_COMM_WORLD, "Dt = %g\n", dt);

  TS  ts;
  Vec x0;
  ierr = MyTsCreate(PETSC_COMM_WORLD, &ts, input_filename, mesh, phys, dt); CHKERRQ(ierr);
  ierr = MeshCreateGlobalVector(mesh, &x0);                                 CHKERRQ(ierr);
  ierr = MeshApplyFunction(mesh, 0, InitialCondition, phys, x0);            CHKERRQ(ierr);
  ierr = TSSolve(ts, x0);                                                   CHKERRQ(ierr);

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
