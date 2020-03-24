#include "spatial.h"
#include "physics.h"

static char mesh_filename[] = "/home/pierre/c/yanss/data/box.msh";  // TODO
static PetscInt overlap = 1;  // TODO: if 0, the balance seems better but the result diverge
static PetscInt buffer_size = 32;  // TODO

PetscErrorCode SetMesh(MPI_Comm comm, DM *mesh, PetscFV *fvm, Physics phys){
  PetscErrorCode ierr;
  PetscInt       dim, dof, i, j;
  DM             foo_dm;
  PetscDS        system;
  char           buffer[buffer_size];

  PetscFunctionBeginUser;
  ierr = DMPlexCreateFromFile(comm, mesh_filename, PETSC_TRUE, mesh); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*mesh, NULL, "-dm_view_orig");             CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(*mesh, PETSC_TRUE, PETSC_FALSE);         CHKERRQ(ierr);
  ierr = DMPlexDistribute(*mesh, overlap, NULL, &foo_dm);             CHKERRQ(ierr);
  if (foo_dm) {
    ierr = DMDestroy(mesh);                                           CHKERRQ(ierr);
    *mesh = foo_dm;
  }
  ierr = DMSetFromOptions(*mesh);                                     CHKERRQ(ierr);
  ierr = DMViewFromOptions(*mesh, NULL, "-dm_view_para");             CHKERRQ(ierr);
  ierr = DMPlexConstructGhostCells(*mesh, NULL, NULL, &foo_dm);       CHKERRQ(ierr);
  ierr = DMDestroy(mesh);                                             CHKERRQ(ierr);
  *mesh = foo_dm;
  ierr = DMViewFromOptions(*mesh, NULL, "-dm_view");                  CHKERRQ(ierr);
  ierr = DMGetDimension(*mesh, &dim);                                 CHKERRQ(ierr);

  ierr = PetscFVCreate(PETSC_COMM_WORLD, fvm);                                          CHKERRQ(ierr);
  ierr = PetscFVSetNumComponents(*fvm, phys->dof);                                      CHKERRQ(ierr);
  ierr = PetscFVSetSpatialDimension(*fvm, dim);                                         CHKERRQ(ierr);
  for (i = 0, dof = 0; i < phys->nfields; i++){
    if (phys->field_desc[i].dof == 1) {
      ierr = PetscFVSetComponentName(*fvm, dof, phys->field_desc[i].name);              CHKERRQ(ierr);
    }
    else {
      for (j=0; j < phys->field_desc[i].dof; j++){
        ierr = PetscSNPrintf(buffer, buffer_size,"%s_%d", phys->field_desc[i].name, j); CHKERRQ(ierr);
        ierr = PetscFVSetComponentName(*fvm, dof + j, buffer);                          CHKERRQ(ierr);
      }
      dof += phys->field_desc[i].dof;
    }
  }
  ierr = PetscFVSetFromOptions(*fvm);                                                   CHKERRQ(ierr);
  ierr = DMAddField(*mesh, NULL, (PetscObject) *fvm);                                   CHKERRQ(ierr);

  ierr = DMCreateDS(*mesh);                                 CHKERRQ(ierr);
  ierr = DMGetDS(*mesh, &system);                           CHKERRQ(ierr);
  ierr = PetscDSSetRiemannSolver(system, 0, phys->riemann); CHKERRQ(ierr);
  ierr = PetscDSSetContext(system, 0, phys);                CHKERRQ(ierr);
  ierr = SetBC(system, phys);                               CHKERRQ(ierr);
  ierr = PetscDSSetFromOptions(system);                     CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GetInitialCondition(DM dm, Vec x, Physics phys){
  PetscErrorCode ierr;
  PetscErrorCode (*func[1]) (PetscInt, PetscReal, const PetscReal*, PetscInt, PetscScalar*, void*);
  void           *ctx[1];

  PetscFunctionBeginUser;
  func[0] = InitialCondition;
  ctx[0]  = (void *) phys;
  ierr    = DMProjectFunction(dm, 0.0, func, ctx, INSERT_ALL_VALUES, x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
