#include "physics.h"
#include "spatial.h"
#include "temporal.h"
#include "input.h"

static const char help[] = "Finite Volume solver\n";

int main(int argc, char **argv){
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help); if (ierr) return ierr;

  if (argc < 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "No input file was given: please use \"%s optionfile\"", argv[0]);
  char *input_filename = argv[1];
  ierr = IOLoadPetscOptions(input_filename); CHKERRQ(ierr);

  PetscBool set;
  char mesh_filename[256];
  ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-mesh", mesh_filename, sizeof(mesh_filename), &set); CHKERRQ(ierr);
  if (!set) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "No mesh file was given: please use the option \"-mesh filename\"");

  DM      mesh;
  Physics phys;
  ierr = MeshLoadFromFile(PETSC_COMM_WORLD, mesh_filename, &mesh); CHKERRQ(ierr);
  ierr = PhysicsCreate(&phys, input_filename, mesh);               CHKERRQ(ierr);

  PetscReal cfl = 0.5; // TODO

  TS  ts;
  ierr = MyTsCreate(PETSC_COMM_WORLD, &ts, input_filename, mesh, phys, cfl); CHKERRQ(ierr);
  ierr = TSSolve(ts, phys->x);                                               CHKERRQ(ierr);

  PetscReal         ftime;
  PetscInt          nsteps;
  TSConvergedReason reason;
  ierr = TSGetSolveTime(ts, &ftime);        CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &nsteps);      CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts, &reason); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %D steps\n", TSConvergedReasons[reason], (double)ftime, nsteps); CHKERRQ(ierr);

  ierr = TSDestroy(&ts);        CHKERRQ(ierr);
  ierr = PhysicsDestroy(&phys); CHKERRQ(ierr);
  ierr = MeshDestroy(&mesh);    CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
