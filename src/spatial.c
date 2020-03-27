#include "spatial.h"
#include "physics.h"

#include <petsc/private/dmlabelimpl.h>
#include <petsc/private/isimpl.h>


static char mesh_filename[] = "/home/pierre/c/yanss/data/box.msh";  // TODO
static PetscInt overlap = 1;  // TODO: if 0, the balance seems better but the result diverge

PetscErrorCode SetMesh(MPI_Comm comm, DM *mesh, PetscFV *fvm, Physics phys){
  PetscErrorCode ierr;
  PetscInt       dim;
  DM             foo_dm;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateFromFile(comm, mesh_filename, PETSC_TRUE, mesh); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*mesh, NULL, "-dm_view_orig");             CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(*mesh, PETSC_TRUE, PETSC_FALSE);         CHKERRQ(ierr);
  ierr = DMPlexDistribute(*mesh, overlap, NULL, &foo_dm);             CHKERRQ(ierr);
  if (foo_dm) {
    ierr = DMDestroy(mesh);                                           CHKERRQ(ierr);
    *mesh = foo_dm;
  }
  ierr = DMViewFromOptions(*mesh, NULL, "-dm_view_para");             CHKERRQ(ierr);
  ierr = DMSetFromOptions(*mesh);                                     CHKERRQ(ierr);
  ierr = DMPlexConstructGhostCells(*mesh, NULL, NULL, &foo_dm);       CHKERRQ(ierr);
  ierr = DMDestroy(mesh);                                             CHKERRQ(ierr);
  *mesh = foo_dm;
  ierr = DMGetDimension(*mesh, &dim);                                 CHKERRQ(ierr);

  ierr = PetscFVCreate(PETSC_COMM_WORLD, fvm);                                          CHKERRQ(ierr);
  ierr = PetscFVSetNumComponents(*fvm, phys->dof);                                      CHKERRQ(ierr);
  ierr = PetscFVSetSpatialDimension(*fvm, dim);                                         CHKERRQ(ierr);
  PetscInt i, dof;
  for (i = 0, dof = 0; i < phys->nfields; i++){
    if (phys->field_desc[i].dof == 1) {
      ierr = PetscFVSetComponentName(*fvm, dof, phys->field_desc[i].name);              CHKERRQ(ierr);
    }
    else {
      PetscInt j;
      for (j=0; j < phys->field_desc[i].dof; j++){
        static PetscInt buffer_size = 32;
        char buffer[buffer_size];
        ierr = PetscSNPrintf(buffer, buffer_size,"%s_%d", phys->field_desc[i].name, j); CHKERRQ(ierr);
        ierr = PetscFVSetComponentName(*fvm, dof + j, buffer);                          CHKERRQ(ierr);
      }
      dof += phys->field_desc[i].dof;
    }
  }
  ierr = PetscFVSetFromOptions(*fvm);                                                   CHKERRQ(ierr);
  ierr = DMAddField(*mesh, NULL, (PetscObject) *fvm);                                   CHKERRQ(ierr);

  PetscDS system;
  ierr = DMCreateDS(*mesh);                                 CHKERRQ(ierr);
  ierr = DMGetDS(*mesh, &system);                           CHKERRQ(ierr);
  ierr = PetscDSSetRiemannSolver(system, 0, phys->riemann); CHKERRQ(ierr);
  ierr = PetscDSSetContext(system, 0, phys);                CHKERRQ(ierr);
  ierr = SetBC(system, phys);                               CHKERRQ(ierr);
  ierr = PetscDSSetFromOptions(system);                     CHKERRQ(ierr);

  char      opt[] = "draw";
  PetscBool flag1, flag2;
  PetscInt  numGhostCells;
  ierr = PetscOptionsGetString(NULL, NULL, "-dm_view", opt, sizeof(opt), &flag1); CHKERRQ(ierr);
  ierr = PetscStrcmp(opt, "draw", &flag2);                                        CHKERRQ(ierr);
  if (flag1 && flag2) {
    ierr = HideGhostCells(*mesh, &numGhostCells);                                 CHKERRQ(ierr);
  }
  ierr = DMViewFromOptions(*mesh, NULL, "-dm_view");                              CHKERRQ(ierr);
  if (flag1 && flag2) {
    ierr = RestoreGhostCells(*mesh, numGhostCells);                               CHKERRQ(ierr);
  }

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

PetscErrorCode HideGhostCells(DM dm, PetscInt *numGhostCells){
  PetscErrorCode ierr;
  DMLabel        label;
  PetscInt       dim, pHyb, pStart, pEnd;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);                           CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &pHyb, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, dim, &pStart, &pEnd);     CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &label);                    CHKERRQ(ierr);
  pEnd -= pStart;
  pHyb -= pStart;
  label->points[dim]->max -= (pEnd - pHyb);
  if (numGhostCells) *numGhostCells = (pEnd - pHyb);
  PetscFunctionReturn(0);
}

PetscErrorCode RestoreGhostCells(DM dm, PetscInt numGhostCells){
  PetscErrorCode ierr;
  DMLabel        label;
  PetscInt       dim;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);        CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &label); CHKERRQ(ierr);
  label->points[dim]->max += numGhostCells;
  PetscFunctionReturn(0);
}
