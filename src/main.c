#include "physics.h"
#include "spatial.h"
#include "temporal.h"
#include "input.h"

static const char help[] = "Finite Volume solver\n";

int main(int argc, char **argv){
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;

  if (argc < 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "No input file was given: please use \"%s optionfile\"", argv[0]);
  char *input_filename = argv[1];
  ierr = YAMLLoadPetscOptions(input_filename); CHKERRQ(ierr);

  PetscBool set;
  char mesh_filename[256];
  ierr = PetscOptionsGetString(NULL, NULL, "-mesh", mesh_filename, sizeof(mesh_filename), &set); CHKERRQ(ierr);
  if (!set) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "No mesh file was given: please use the option \"-mesh filename\"");

  DM mesh;
  ierr = MeshLoadFromFile(PETSC_COMM_WORLD, mesh_filename, input_filename, &mesh); CHKERRQ(ierr);

  Physics phys;
  ierr = PhysicsCreate(&phys, input_filename, mesh); CHKERRQ(ierr);

  ierr = MeshSetUp(mesh, phys, input_filename); CHKERRQ(ierr);

  Vec x;
  ierr = MeshCreateGlobalVector(mesh, &x);                      CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");       CHKERRQ(ierr);
  ierr = MeshApplyFunction(mesh, 0, InitialCondition, phys, x); CHKERRQ(ierr);

  TS ts;
  ierr = TSCreate_User(PETSC_COMM_WORLD, &ts, input_filename, mesh, phys); CHKERRQ(ierr);
  ierr = TSSolve(ts, x);                                                   CHKERRQ(ierr);

  PetscReal         ftime;
  PetscInt          nsteps;
  TSConvergedReason reason;
  ierr = TSGetSolveTime(ts, &ftime);        CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &nsteps);      CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts, &reason); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %D steps\n", TSConvergedReasons[reason], (double)ftime, nsteps); CHKERRQ(ierr);

  ierr = VecDestroy(&x);        CHKERRQ(ierr);
  ierr = TSDestroy(&ts);        CHKERRQ(ierr);
  ierr = PhysicsDestroy(&phys); CHKERRQ(ierr);
  ierr = MeshDestroy(&mesh);    CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
