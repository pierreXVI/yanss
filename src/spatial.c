#include "spatial.h"
#include "physics.h"
#include "input.h"
#include "view.h"


/*
  Fills the periodicity context
  disp is the displacement from bc_1 to bc_2
*/
static PetscErrorCode MeshSetUp_PerioCtx_Ctx(DM dm, PetscInt bc_1, PetscInt bc_2, PetscReal *disp, struct PerioCtx *ctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscInt Nc, dim, bcStart, bcEnd;
  { // Get problem related values
    PetscFV  fvm;
    ierr = DMGetDimension(dm, &dim);                             CHKERRQ(ierr);
    ierr = MeshGetCellStratum(dm, NULL, NULL, &bcStart, &bcEnd); CHKERRQ(ierr);
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);         CHKERRQ(ierr);
    ierr = PetscFVGetNumComponents(fvm, &Nc);                    CHKERRQ(ierr);
  }

  PetscInt       nface1_loc, nface2_loc, nface_loc;
  IS             face1_is, face2_is;
  const PetscInt *face1, *face2;
  { // Get face description
    ierr = DMGetStratumIS(dm, "Face Sets", bc_1, &face1_is); CHKERRQ(ierr);
    ierr = DMGetStratumIS(dm, "Face Sets", bc_2, &face2_is); CHKERRQ(ierr);
    if (!face1_is) {ierr = ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_OWN_POINTER, &face1_is); CHKERRQ(ierr);}
    if (!face2_is) {ierr = ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_OWN_POINTER, &face2_is); CHKERRQ(ierr);}
    ierr = ISGetSize(face1_is, &nface1_loc); CHKERRQ(ierr);
    ierr = ISGetSize(face2_is, &nface2_loc); CHKERRQ(ierr);
    ierr = ISGetIndices(face1_is, &face1);   CHKERRQ(ierr);
    ierr = ISGetIndices(face2_is, &face2);   CHKERRQ(ierr);

    for (PetscInt i = 0; i < nface1_loc; i++) {
      PetscInt support_size;
      ierr = DMPlexGetSupportSize(dm, face1[i], &support_size); CHKERRQ(ierr);
      if (support_size != 2) nface1_loc = i;
    }
    for (PetscInt j = 0; j < nface2_loc; j++) {
      PetscInt support_size;
      ierr = DMPlexGetSupportSize(dm, face2[j], &support_size); CHKERRQ(ierr);
      if (support_size != 2) nface2_loc = j;
    }
    nface_loc = nface1_loc + nface2_loc;
  }

  PetscInt        block_start;
  Vec             coord_vec;
  const PetscReal *coord;
  PetscInt  *master_array, *slave_array;
  { // Get face coordinates and support
    Vec        coord_mpi;
    VecScatter scatter;
    ierr = PetscMalloc1(nface_loc, &master_array);                   CHKERRQ(ierr);
    ierr = PetscMalloc1(nface_loc, &slave_array);                    CHKERRQ(ierr);
    ierr = VecCreate(PetscObjectComm((PetscObject) dm), &coord_mpi); CHKERRQ(ierr);
    ierr = VecSetType(coord_mpi, VECMPI);                            CHKERRQ(ierr);
    ierr = VecSetSizes(coord_mpi, nface_loc * dim, PETSC_DECIDE);    CHKERRQ(ierr);
    ierr = VecSetBlockSize(coord_mpi, dim);                          CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(coord_mpi, &block_start, NULL);      CHKERRQ(ierr);
    block_start /= dim;
    for (PetscInt i = 0; i < nface1_loc; i++) {
      PetscInt  loc;
      PetscReal val[dim];
      loc = i + block_start;
      ierr = DMPlexComputeCellGeometryFVM(dm, face1[i], NULL, val, NULL); CHKERRQ(ierr);
      ierr = VecSetValuesBlocked(coord_mpi, 1, &loc, val, INSERT_VALUES); CHKERRQ(ierr);

      const PetscInt *support;
      ierr = DMPlexGetSupport(dm, face1[i], &support);                    CHKERRQ(ierr);
      if (bcStart <= support[0] && support[0] < bcEnd) {
        master_array[i] = support[1];
        slave_array[i] = support[0];
      } else {
        master_array[i] = support[0];
        slave_array[i] = support[1];
      }
    }
    for (PetscInt j = 0; j < nface2_loc; j++) {
      PetscInt  loc;
      PetscReal val[dim];
      loc = j + block_start + nface1_loc;
      ierr = DMPlexComputeCellGeometryFVM(dm, face2[j], NULL, val, NULL); CHKERRQ(ierr);
      ierr = VecSetValuesBlocked(coord_mpi, 1, &loc, val, INSERT_VALUES); CHKERRQ(ierr);

      const PetscInt *support;
      ierr = DMPlexGetSupport(dm, face2[j], &support);                    CHKERRQ(ierr);
      if (bcStart <= support[0] && support[0] < bcEnd) {
        master_array[nface1_loc + j] = support[1];
        slave_array[nface1_loc + j] = support[0];
      } else {
        master_array[nface1_loc + j] = support[0];
        slave_array[nface1_loc + j] = support[1];
      }
    }
    ierr = VecAssemblyBegin(coord_mpi);                                                     CHKERRQ(ierr);
    ierr = VecAssemblyEnd(coord_mpi);                                                       CHKERRQ(ierr);
    ierr = VecScatterCreateToAll(coord_mpi, &scatter, &coord_vec);                          CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter, coord_mpi, coord_vec, INSERT_VALUES, SCATTER_FORWARD);  CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, coord_mpi, coord_vec, INSERT_VALUES, SCATTER_FORWARD);    CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter);                                                     CHKERRQ(ierr);
    ierr = VecDestroy(&coord_mpi);                                                          CHKERRQ(ierr);
    ierr = VecGetArrayRead(coord_vec, &coord);                                              CHKERRQ(ierr);
  }

  PetscInt nface_tot;
  ierr = VecGetSize(coord_vec, &nface_tot); CHKERRQ(ierr);
  nface_tot /= dim;

  PetscInt *localToGlobal;
  ierr = PetscMalloc1(nface_loc, &localToGlobal); CHKERRQ(ierr);

  for (PetscInt i = 0; i < nface1_loc; i++) {
    PetscBool found = PETSC_FALSE;
    PetscReal len;
    ierr = DMPlexComputeCellGeometryFVM(dm, face1[i], &len, NULL, NULL); CHKERRQ(ierr);
    for (PetscInt j = 0; j < nface_tot; j++) {
      PetscReal dist = 0;
      for (PetscInt k = 0; k < dim; k++) dist += PetscSqr(coord[dim * j + k] - coord[dim * (block_start + i) + k] - disp[k]);
      if (PetscSqrtReal(dist) / len < 1E-1) {
        found = PETSC_TRUE;
        localToGlobal[i] = j;
        break;
      }
    }
    if (!found) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Cannot find periodic face on boundary %d for face %d on boundary %d with given displacement", bc_2, face1[i], bc_1);
  }
  for (PetscInt j = 0; j < nface2_loc; j++) {
    PetscBool found = PETSC_FALSE;
    PetscReal len;
    ierr = DMPlexComputeCellGeometryFVM(dm, face2[j], &len, NULL, NULL); CHKERRQ(ierr);
    for (PetscInt i = 0; i < nface_tot; i++) {
      PetscReal dist = 0;
      for (PetscInt k = 0; k < dim; k++) dist += PetscSqr(coord[dim * (block_start + nface1_loc + j) + k] - coord[dim * i + k] - disp[k]);
      if (PetscSqrtReal(dist) / len < 1E-1) {
        found = PETSC_TRUE;
        localToGlobal[nface1_loc + j] = i;
        break;
      }
    }
    if (!found) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Cannot find periodic face on boundary %d for face %d on boundary %d with given displacement", bc_1, face2[j], bc_2);
  }

  ierr = ISRestoreIndices(face1_is, &face1);     CHKERRQ(ierr);
  ierr = ISRestoreIndices(face2_is, &face2);     CHKERRQ(ierr);
  ierr = ISDestroy(&face1_is);                   CHKERRQ(ierr);
  ierr = ISDestroy(&face2_is);                   CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coord_vec, &coord); CHKERRQ(ierr);
  ierr = VecDestroy(&coord_vec);                 CHKERRQ(ierr);

  ISLocalToGlobalMapping mapping;
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nface_loc, master_array, PETSC_OWN_POINTER, &ctx->master); CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nface_loc, slave_array, PETSC_OWN_POINTER, &ctx->slave);   CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, Nc * (dim + 1), nface_loc, localToGlobal, PETSC_OWN_POINTER, &mapping); CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) dm), &ctx->buffer);                                 CHKERRQ(ierr);
  ierr = VecSetType(ctx->buffer, VECMPI);                                                            CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->buffer, nface_loc * Nc * (dim + 1), PETSC_DECIDE);                         CHKERRQ(ierr);
  ierr = VecSetBlockSize(ctx->buffer, Nc * (dim + 1));                                               CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(ctx->buffer, mapping);                                           CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&mapping);                                                    CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
  Reads the periodicity from the input file, and construct the `perio` context array
  The periodicity contexts can only be created after some of the physical context is filled, as the number of components is needed
*/
static PetscErrorCode MeshSetUp_PerioCtx(DM dm, const char *opt_filename){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscInt       dim, num, num_loc; // Mesh dimension, number of boundaries on mesh and on local process
  IS             bnd_is, bnd_is_loc;
  const PetscInt *bnd, *bnd_loc;
  {
    IS bnd_is_mpi;
    ierr = DMGetDimension(dm, &dim);                                                                 CHKERRQ(ierr);
    ierr = DMGetLabelIdIS(dm, "Face Sets", &bnd_is_loc);                                             CHKERRQ(ierr);
    ierr = ISOnComm(bnd_is_loc, PetscObjectComm((PetscObject) dm), PETSC_USE_POINTER , &bnd_is_mpi); CHKERRQ(ierr);
    ierr = ISAllGather(bnd_is_mpi, &bnd_is);                                                         CHKERRQ(ierr);
    ierr = ISDestroy(&bnd_is_mpi);                                                                   CHKERRQ(ierr);
    ierr = ISSortRemoveDups(bnd_is);                                                                 CHKERRQ(ierr);
    ierr = ISGetSize(bnd_is, &num);                                                                  CHKERRQ(ierr);
    ierr = ISGetSize(bnd_is_loc, &num_loc);                                                          CHKERRQ(ierr);
    ierr = ISGetIndices(bnd_is_loc, &bnd_loc);                                                       CHKERRQ(ierr);
    ierr = ISGetIndices(bnd_is, &bnd);                                                               CHKERRQ(ierr);
  }

  MeshCtx ctx;
  ierr = DMGetApplicationContext(dm, &ctx); CHKERRQ(ierr);

  PetscBool       rem[num_loc]; // Boundaries to remove
  struct PerioCtx ctxs[num / 2]; // Periodicity contexts
  {
    ctx->n_perio = 0;
    for (PetscInt i = 0; i < num_loc; i++) rem[i] = PETSC_FALSE;
    for (PetscInt i = 0; i < num; i++) {
      PetscInt master;
      PetscReal *disp;
      ierr = YAMLLoadPeriodicity(opt_filename, bnd[i], dim, &master, &disp); CHKERRQ(ierr);
      if (disp) {
        PetscInt i_master, i_slave;
        ierr = ISLocate(bnd_is_loc, master, &i_master); CHKERRQ(ierr);
        ierr = ISLocate(bnd_is_loc, bnd[i], &i_slave);  CHKERRQ(ierr);
        if (i_master >= 0) rem[i_master] = PETSC_TRUE;
        if (i_slave >= 0) rem[i_slave] = PETSC_TRUE;
        ierr = MeshSetUp_PerioCtx_Ctx(dm, master, bnd[i], disp, &ctxs[ctx->n_perio++]); CHKERRQ(ierr);
        ierr = PetscFree(disp);                                                         CHKERRQ(ierr);
      }
    }
  }

  { // Removing periodic boundaries from "Face Sets"
    DMLabel label_old, label_new;
    ierr = DMRemoveLabel(dm, "Face Sets", &label_old); CHKERRQ(ierr);
    ierr = DMCreateLabel(dm, "Face Sets");             CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "Face Sets", &label_new);    CHKERRQ(ierr);
    for (PetscInt i = 0; i < num_loc; i++) {
      if (!rem[i]) {
        IS points;
        ierr = DMLabelGetStratumIS(label_old, bnd_loc[i], &points); CHKERRQ(ierr);
        ierr = DMLabelSetStratumIS(label_new, bnd_loc[i], points);  CHKERRQ(ierr);
        ierr = ISDestroy(&points);                                  CHKERRQ(ierr);
      }
    }
    ierr = DMLabelDestroy(&label_old); CHKERRQ(ierr);
  }

  { // Cleanup
    ierr = ISRestoreIndices(bnd_is_loc, &bnd_loc); CHKERRQ(ierr);
    ierr = ISRestoreIndices(bnd_is, &bnd);         CHKERRQ(ierr);
    ierr = ISDestroy(&bnd_is_loc);                 CHKERRQ(ierr);
    ierr = ISDestroy(&bnd_is);                     CHKERRQ(ierr);
  }

  { // Setting periodicity context for the mesh
    ierr = PetscMalloc1(ctx->n_perio, &ctx->perio); CHKERRQ(ierr);
    for (PetscInt n = 0; n < ctx->n_perio; n++) ctx->perio[n] = ctxs[n];
  }

  PetscFunctionReturn(0);
}


/*
  Sets up the mesh context:
    - stores the cell indices
    - compute geometric factors for cell centered gradient reconstruction
    - compute geometric factors for face centered gradient reconstruction
  Only the periodicity context is not stored, as it needs the physical context.
*/
static PetscErrorCode MeshSetUp_Ctx(DM dm){
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscFV         fvm;
  Vec             cellgeom, facegeom;
  MeshCtx         ctx;
  DM              dmCell, dmFace;
  PetscReal       *dx;
  const PetscReal *cellgeom_a, *facegeom_a;
  PetscInt        *neighbors, dim, ncell, fStart, fEnd;
  { // Getting mesh data
    PetscInt maxNumNeighbors, nface = 0, ghost;

    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);                          CHKERRQ(ierr);
    ierr = DMPlexGetDataFVM(dm, fvm, &cellgeom, &facegeom, NULL);                 CHKERRQ(ierr);
    ierr = VecGetDM(cellgeom, &dmCell);                                           CHKERRQ(ierr);
    ierr = VecGetDM(facegeom, &dmFace);                                           CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom, &cellgeom_a);                                CHKERRQ(ierr);
    ierr = VecGetArrayRead(facegeom, &facegeom_a);                                CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);                                              CHKERRQ(ierr);

    ierr = PetscNew(&ctx);                                                        CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);                         CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &(ctx->cStartCell), &(ctx->cEnd));       CHKERRQ(ierr);
    ierr = DMPlexGetGhostCellStratum(dm, &(ctx->cStartBoundary), NULL);           CHKERRQ(ierr);
    for (PetscInt f = fStart; f < fEnd; f++) {
      ierr = DMGetLabelValue(dm, "ghost", f, &ghost);                             CHKERRQ(ierr);
      if (ghost < 0) nface++;
    }
    for (ctx->cStartOverlap = ctx->cStartCell; ctx->cStartOverlap < ctx->cStartBoundary; ctx->cStartOverlap++) {
      ierr = DMGetLabelValue(dm, "ghost", ctx->cStartOverlap, &ghost);            CHKERRQ(ierr);
      if (ghost >= 0) break;
    }
    ncell = ctx->cStartOverlap - ctx->cStartCell;
    ierr = PetscMalloc2(ncell, &ctx->CellCtx, nface, &ctx->FaceCtx);              CHKERRQ(ierr);

    ierr = DMPlexGetMaxSizes(dm, &maxNumNeighbors, NULL);                         CHKERRQ(ierr);
    maxNumNeighbors = PetscSqr(maxNumNeighbors);
    ierr = PetscFVLeastSquaresSetMaxFaces(fvm, maxNumNeighbors);                  CHKERRQ(ierr);
    ierr = PetscMalloc2(maxNumNeighbors, &neighbors, maxNumNeighbors * dim, &dx); CHKERRQ(ierr);
  }

  for (PetscInt n = 0; n < ncell; n++) { // Getting neighbors
    PetscInt       nface, nface1, c = ctx->cStartCell + n, nc, size, numNeighbors = 0;
    const PetscInt *faces, *faces1, *fcells;
    ierr = DMPlexGetConeSize(dm, c, &nface); CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &faces);     CHKERRQ(ierr);
    for (PetscInt f = 0; f < nface; f++) {
      ierr = DMPlexGetSupport(dm, faces[f], &fcells); CHKERRQ(ierr);
      nc = (c == fcells[0]) ? fcells[1] : fcells[0];
      if (nc >= ctx->cStartBoundary) continue;
      neighbors[numNeighbors++] = nc;
      ierr = DMPlexGetConeSize(dm, nc, &nface1); CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, nc, &faces1);     CHKERRQ(ierr);
      for (PetscInt f1 = 0; f1 < nface1; f1++) {
        ierr = DMPlexGetSupportSize(dm, faces1[f1], &size); CHKERRQ(ierr);
        if (size != 2) continue;
        ierr = DMPlexGetSupport(dm, faces1[f1], &fcells); CHKERRQ(ierr);
        if (fcells[0] != c && fcells[0] != nc && fcells[0] < ctx->cStartBoundary) neighbors[numNeighbors++] = fcells[0];
        if (fcells[1] != c && fcells[1] != nc && fcells[1] < ctx->cStartBoundary) neighbors[numNeighbors++] = fcells[1];
      }
    }

    ierr = ISCreateGeneral(PETSC_COMM_SELF, numNeighbors, neighbors, PETSC_COPY_VALUES, &ctx->CellCtx[n].neighborhood); CHKERRQ(ierr);
    ierr = ISSortRemoveDups(ctx->CellCtx[n].neighborhood);                                                              CHKERRQ(ierr);
    ierr = PetscMalloc1(numNeighbors * dim, &ctx->CellCtx[n].grad_coeff);                                               CHKERRQ(ierr);
  }

  for (PetscInt n = 0; n < ncell; n++) { // Computing gradient coefficients
    PetscInt        numNeighbors;
    const PetscInt  *neighborhood;
    PetscFVCellGeom *cg, *ncg;

    ierr = ISGetSize(ctx->CellCtx[n].neighborhood, &numNeighbors);             CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->CellCtx[n].neighborhood, &neighborhood);          CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, ctx->cStartCell + n, cellgeom_a, &cg); CHKERRQ(ierr);

    for (PetscInt i = 0; i < numNeighbors; i++) {
      ierr = DMPlexPointLocalRead(dmCell, neighborhood[i], cellgeom_a, &ncg); CHKERRQ(ierr);
      for (PetscInt j = 0; j < dim; j++) dx[i * dim + j] = ncg->centroid[j] - cg->centroid[j];
    }
    ierr = PetscFVComputeGradient(fvm, numNeighbors, dx, ctx->CellCtx[n].grad_coeff); CHKERRQ(ierr);
  }

  for (PetscInt f = fStart, iface=0; f < fEnd; f++) {
    PetscInt ghost;
    ierr = DMGetLabelValue(dm, "ghost", f, &ghost); CHKERRQ(ierr);
    if (ghost >= 0) continue;

    const PetscInt  *cells;
    PetscFVFaceGeom *fg;
    PetscFVCellGeom *cg0, *cg1;
    ierr = DMPlexGetSupport(dm, f, &cells);                          CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmFace, f, facegeom_a, &fg);         CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom_a, &cg0); CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom_a, &cg1); CHKERRQ(ierr);

    PetscReal dx0 = 0, dx1 = 0;
    for (size_t k = 0; k < dim; k++) dx0 += PetscSqr(fg->centroid[k] - cg0->centroid[k]);
    for (size_t k = 0; k < dim; k++) dx1 += PetscSqr(fg->centroid[k] - cg1->centroid[k]);
    ctx->FaceCtx[iface] = PetscSqr(dx0 / dx1);
    iface++;
  }

  { // Cleanup
    ierr = VecRestoreArrayRead(facegeom, &facegeom_a); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellgeom, &cellgeom_a); CHKERRQ(ierr);
    ierr = PetscFree2(neighbors, dx);                  CHKERRQ(ierr);
  }

  ierr = DMSetApplicationContext(dm, ctx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*
  Compute the RHS
*/
static PetscErrorCode MeshComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *ctx){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  Vec locF;
  ierr = DMGetLocalVector(dm, &locF); CHKERRQ(ierr);
  ierr = VecZeroEntries(locF);        CHKERRQ(ierr);

  PetscFV   fvm;
  PetscInt  fStart, fEnd, cStart, cEndCell, cEnd, Nc;
  PetscReal *locX_array;
  { // Getting mesh data
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);           CHKERRQ(ierr);
    ierr = MeshGetCellStratum(dm, &cStart, &cEndCell, &cEnd, NULL); CHKERRQ(ierr);
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);            CHKERRQ(ierr);
    ierr = PetscFVGetNumComponents(fvm, &Nc);                       CHKERRQ(ierr);
    ierr = VecGetArray(locX, &locX_array);                          CHKERRQ(ierr);
  }

  { // Convertion conservative to primitive
    // // For each "true" cell and each overlapping cell (the MPI exchanges are done before, on conservative variables)
    // for (PetscInt c = cStart; c < cEnd; c++) {
    //   PetscReal *valX;
    //   ierr = DMPlexPointLocalFieldRef(dm, c, 0, locX_array, &valX); CHKERRQ(ierr);
    //   ConservativeToPrimitive(valX, valX, ctx);
    // }
    VecApplyFunctionInPlace(locX, ConservativeToPrimitive, ctx);
  }

  DM  dmGrad;
  Vec locGrad, cellgeom, facegeom;
  { // Computes the gradient, fills Neumann boundaries
    Vec grad;
    ierr = DMPlexGetDataFVM(dm, fvm, &cellgeom, &facegeom, &dmGrad);                                 CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dmGrad, &grad);                                                         CHKERRQ(ierr);
    ierr = MeshReconstructGradientsFVM(dm, locX, grad);                                              CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmGrad, &locGrad);                                                       CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad);                               CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad);                                 CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmGrad, &grad);                                                     CHKERRQ(ierr);
    ierr = MeshInsertPeriodicValues(dm, locX, locGrad);                                              CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, time, facegeom, cellgeom, locGrad);     CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(dmGrad, PETSC_FALSE, locGrad, time, facegeom, cellgeom, NULL); CHKERRQ(ierr);
  }

  PetscFVFaceGeom *fgeom;
  PetscReal       *vol, *uL, *uR, *fluxL, *fluxR;
  { // Extracting fields, Riemann solve
    PetscDS  prob;
    PetscInt numFaces;
    ierr = DMGetDS(dm, &prob);                                                                                  CHKERRQ(ierr);
    ierr = DMPlexGetFaceFields(dm, fStart, fEnd, locX, NULL, facegeom, cellgeom, locGrad, &numFaces, &uL, &uR); CHKERRQ(ierr);
    ierr = DMPlexGetFaceGeometry(dm, fStart, fEnd, facegeom, cellgeom, &numFaces, &fgeom, &vol);                CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, numFaces * Nc, MPIU_REAL, &fluxL);                                                CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, numFaces * Nc, MPIU_REAL, &fluxR);                                                CHKERRQ(ierr);
    ierr = PetscArrayzero(fluxL, numFaces * Nc);                                                                CHKERRQ(ierr);
    ierr = PetscArrayzero(fluxR, numFaces * Nc);                                                                CHKERRQ(ierr);
    ierr = PetscFVIntegrateRHSFunction(fvm, prob, 0, numFaces, fgeom, vol, uL, uR, fluxL, fluxR);               CHKERRQ(ierr);

    {
    //   void            (*riemann)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[], PetscReal[], void *);
    //   void            *rctx;
    //   PetscReal        flux[Nc];
    //   const PetscReal *constants;
    //   PetscInt         dim, numConstants, pdim, Nc, totDim, off, f, d;
    //   PetscErrorCode   ierr;
    //
    //   ierr = PetscDSGetTotalComponents(prob, &Nc);CHKERRQ(ierr);
    //   ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
    //   ierr = PetscDSGetFieldOffset(prob, 0, &off);CHKERRQ(ierr);
    //   ierr = PetscDSGetRiemannSolver(prob, 0, &riemann);CHKERRQ(ierr);
    //   ierr = PetscDSGetContext(prob, 0, &rctx);CHKERRQ(ierr);
    //   ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
    //   ierr = PetscFVGetSpatialDimension(fvm, &dim);CHKERRQ(ierr);
    //   ierr = PetscFVGetNumComponents(fvm, &pdim);CHKERRQ(ierr);
    //   for (f = 0; f < numFaces; ++f) {
    //     PetscReal fReal = f;
    //     constants = &fReal;
    //     (*riemann)(dim, pdim, fgeom[f].centroid, fgeom[f].normal, &uL[f*Nc], &uR[f*Nc], numConstants, constants, flux, rctx);
    //     for (d = 0; d < pdim; ++d) {
    //       fluxL[f*totDim+off+d] = flux[d] / vol[f*2+0];
    //       fluxR[f*totDim+off+d] = flux[d] / vol[f*2+1];
    //     }
    //   }
    }

    {
    //   PetscInt        dim;
    //   MeshCtx         mesh_ctx;
    //   const PetscReal *grad_a;
    //   ierr = DMGetDimension(dm, &dim);               CHKERRQ(ierr);
    //   ierr = DMGetApplicationContext(dm, &mesh_ctx); CHKERRQ(ierr);
    //   ierr = VecGetArrayRead(locGrad, &grad_a);      CHKERRQ(ierr);
    //
    //   for (PetscInt f = fStart, iface=0; f < fEnd; f++) { // Computing face gradient
    //     PetscInt ghost;
    //     ierr = DMGetLabelValue(dm, "ghost", f, &ghost);     CHKERRQ(ierr);
    //     if (ghost >= 0) continue;
    //
    //     const PetscInt *cells;
    //     PetscReal      *g0, *g1;
    //     ierr = DMPlexGetSupport(dm, f, &cells);                     CHKERRQ(ierr);
    //     ierr = DMPlexPointLocalRead(dmGrad, cells[0], grad_a, &g0); CHKERRQ(ierr);
    //     ierr = DMPlexPointLocalRead(dmGrad, cells[1], grad_a, &g1); CHKERRQ(ierr);
    //
    //     PetscReal face_grad[Nc * dim];
    //     for (PetscInt k = 0; k < dim * Nc; k++) face_grad[k] = (g0[k] + mesh_ctx->FaceCtx[iface] * g1[k]) / (1 + mesh_ctx->FaceCtx[iface]);
    //
    //     for (PetscInt j = 0; j < Nc; j++) {
    //       PetscReal foo = 0;
    //       for (PetscInt k = 0; k < dim; k++) foo += fgeom[iface].normal[k] * face_grad[j * Nc + k];
    //       fluxL[iface * Nc + j] += -5*foo;
    //       fluxR[iface * Nc + j] += -5*foo;
    //
    //     }
    //     iface++;
    //   }
    //   ierr = VecRestoreArrayRead(locGrad, &grad_a); CHKERRQ(ierr);
    }
  }

  { // Filling flux vector
    PetscReal *fa;
    ierr = VecGetArray(locF, &fa); CHKERRQ(ierr);

    for (PetscInt f = fStart, iface = 0; f < fEnd; f++) {
      const PetscInt *cells;
      PetscInt       ghost;
      PetscReal      *fL = NULL, *fR = NULL;

      ierr = DMGetLabelValue(dm, "ghost", f, &ghost); CHKERRQ(ierr);
      if (ghost >= 0) continue;

      ierr = DMPlexGetSupport(dm, f, &cells);                                              CHKERRQ(ierr);
      if (cells[0] < cEndCell) {ierr = DMPlexPointLocalFieldRef(dm, cells[0], 0, fa, &fL); CHKERRQ(ierr);}
      if (cells[1] < cEndCell) {ierr = DMPlexPointLocalFieldRef(dm, cells[1], 0, fa, &fR); CHKERRQ(ierr);}
      for (PetscInt i = 0; i < Nc; i++) {
        if (fL) fL[i] -= fluxL[iface * Nc + i];
        if (fR) fR[i] += fluxR[iface * Nc + i];
      }
      iface++;
    }
    ierr = VecRestoreArray(locF, &fa); CHKERRQ(ierr);
  }

  { // Convertion primitive to conservative
    // For each "true" cell and each overlapping cell (the MPI exchanges are done before, on conservative variables)
    for (PetscInt c = cStart; c < cEndCell; c++) {
      PetscReal *valX;
      ierr = DMPlexPointLocalFieldRef(dm, c, 0, locX_array, &valX); CHKERRQ(ierr);
      PrimitiveToConservative(valX, valX, ctx);
    }
  }

  { // Cleanup
    ierr = VecRestoreArray(locX, &locX_array);                                                                 CHKERRQ(ierr);
    ierr = DMPlexRestoreFaceFields(dm, fStart, fEnd, locX, NULL, facegeom, cellgeom, locGrad, NULL, &uL, &uR); CHKERRQ(ierr);
    ierr = DMPlexRestoreFaceGeometry(dm, fStart, fEnd, facegeom, cellgeom, NULL, &fgeom, &vol);                CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, 0, MPIU_REAL, &fluxL);                                                       CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, 0, MPIU_REAL, &fluxR);                                                       CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmGrad, &locGrad);                                                             CHKERRQ(ierr);

    ierr = DMLocalToGlobalBegin(dm, locF, ADD_VALUES, F); CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, locF, ADD_VALUES, F);   CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locF);               CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}



PetscErrorCode MeshDestroy(DM *dm){
  PetscErrorCode ierr;
  PetscFV        fvm, fvmGrad;
  DM             dmGrad;
  MeshCtx        ctx;

  PetscFunctionBeginUser;
  ierr = DMGetField(*dm, 0, NULL, (PetscObject*) &fvm);        CHKERRQ(ierr);
  ierr = DMPlexGetDataFVM(*dm, fvm, NULL, NULL, &dmGrad);      CHKERRQ(ierr);
  ierr = DMGetField(dmGrad, 0, NULL, (PetscObject*) &fvmGrad); CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fvmGrad);                             CHKERRQ(ierr);

  ierr = PetscFVDestroy(&fvm);                                 CHKERRQ(ierr);
  ierr = DMGetApplicationContext(*dm, &ctx);                   CHKERRQ(ierr);
  for (PetscInt n = 0; n < ctx->n_perio; n++) {
    ierr = VecDestroy(&ctx->perio[n].buffer);                  CHKERRQ(ierr);
    ierr = ISDestroy(&ctx->perio[n].master);                   CHKERRQ(ierr);
    ierr = ISDestroy(&ctx->perio[n].slave);                    CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->perio);                                CHKERRQ(ierr);
  for (PetscInt n = 0; n < ctx->cStartOverlap - ctx->cStartCell; n++) {
    ierr = ISDestroy(&ctx->CellCtx[n].neighborhood);           CHKERRQ(ierr);
    ierr = PetscFree(ctx->CellCtx[n].grad_coeff);              CHKERRQ(ierr);
  }
  ierr = PetscFree2(ctx->CellCtx, ctx->FaceCtx);               CHKERRQ(ierr);
  ierr = PetscFree(ctx);                                       CHKERRQ(ierr);
  ierr = DMDestroy(dm);                                        CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MeshLoadFromFile(MPI_Comm comm, const char *filename, const char *opt_filename, DM *dm){
  PetscErrorCode ierr;
  PetscInt       dim;
  DM             foo_dm;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm); CHKERRQ(ierr);
  ierr = DMGetDimension(*dm, &dim);                            CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-mesh_view_orig");      CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(*dm, PETSC_TRUE, PETSC_FALSE);    CHKERRQ(ierr);

  { // Partitioning
    PetscPartitioner part;
    ierr = DMPlexGetPartitioner(*dm, &part);        CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);    CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 2, NULL, &foo_dm); CHKERRQ(ierr);
    if (foo_dm) {
      ierr = DMDestroy(dm);                         CHKERRQ(ierr);
      *dm = foo_dm;
    }
  }

  ierr = DMSetFromOptions(*dm); CHKERRQ(ierr);

  { // Boundaries
    ierr = DMPlexConstructGhostCells(*dm, NULL, NULL, &foo_dm); CHKERRQ(ierr);
    ierr = DMDestroy(dm);                                       CHKERRQ(ierr);
    *dm = foo_dm;
  }

  { // Setting PetscFV
    PetscFV      fvm;
    PetscLimiter limiter;
    ierr = PetscFVCreate(comm, &fvm);                         CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fvm, "FV Model"); CHKERRQ(ierr);
    ierr = PetscFVSetType(fvm, PETSCFVLEASTSQUARES);          CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvm, dim);              CHKERRQ(ierr);
    ierr = PetscFVGetLimiter(fvm, &limiter);                  CHKERRQ(ierr);
    ierr = PetscLimiterSetType(limiter, PETSCLIMITERNONE);    CHKERRQ(ierr);
    ierr = PetscFVSetFromOptions(fvm);                        CHKERRQ(ierr);
    ierr = DMAddField(*dm, NULL, (PetscObject) fvm);          CHKERRQ(ierr);
  }

  ierr = MeshSetViewer(*dm);                            CHKERRQ(ierr);
  ierr = MeshSetUp_Ctx(*dm);                            CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh"); CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-mesh_view");    CHKERRQ(ierr);

  { // Setting gradient
    DM      dmGrad;
    PetscFV fvmGrad;

    ierr = PetscObjectQuery((PetscObject) *dm, "DMPlex_dmgrad_fvm", (PetscObject*) &dmGrad); CHKERRQ(ierr);
    ierr = DMDestroy(&dmGrad);                                                               CHKERRQ(ierr);
    ierr = DMClone(*dm, &dmGrad);                                                            CHKERRQ(ierr);
    ierr = PetscFVCreate(comm, &fvmGrad);                                                    CHKERRQ(ierr);
    ierr = PetscFVSetSpatialDimension(fvmGrad, dim);                                         CHKERRQ(ierr);
    ierr = DMAddField(dmGrad, NULL, (PetscObject) fvmGrad);                                  CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) *dm, "DMPlex_dmgrad_fvm", (PetscObject) dmGrad); CHKERRQ(ierr);
    ierr = DMDestroy(&dmGrad);                                                               CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}



PetscErrorCode MeshGetCellStratum(DM dm, PetscInt *cStartCell, PetscInt *cStartOverlap, PetscInt *cStartBoundary, PetscInt *cEnd){
  PetscErrorCode ierr;
  MeshCtx  ctx;

  PetscFunctionBeginUser;
  ierr = DMGetApplicationContext(dm, &ctx); CHKERRQ(ierr);
  if (cStartCell) *cStartCell = ctx->cStartCell;
  if (cStartOverlap) *cStartOverlap = ctx->cStartOverlap;
  if (cStartBoundary) *cStartBoundary = ctx->cStartBoundary;
  if (cEnd) *cEnd = ctx->cEnd;
  PetscFunctionReturn(0);
}


PetscErrorCode MeshApplyFunction(DM dm, PetscReal time,
                                 PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal*, PetscInt, PetscReal*, void*),
                                 void *ctx, Vec x){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMProjectFunction(dm, time, &func, &ctx, INSERT_ALL_VALUES, x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MeshCreateGlobalVector(DM dm, Vec *x){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dm, x);                                    CHKERRQ(ierr);
  ierr = VecSetOperation(*x, VECOP_VIEW, (void (*)(void)) VecView_Mesh); CHKERRQ(ierr);

  PetscFV  fvm;
  PetscInt Nc;
  ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm); CHKERRQ(ierr);
  ierr = PetscFVGetNumComponents(fvm, &Nc);            CHKERRQ(ierr);
  ierr = VecSetBlockSize(*x, Nc);                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MeshSetUp(DM dm, Physics phys, const char *filename){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = DMTSSetRHSFunctionLocal(dm, MeshComputeRHSFunctionFVM, phys); CHKERRQ(ierr);

  ierr = MeshSetUp_PerioCtx(dm, filename); CHKERRQ(ierr);

  { // Setting discrete system (Riemann solver, boundaries)
    DM             dmGrad;
    PetscFV        fvm;
    PetscDS        prob, probGrad;
    DMLabel        label;
    IS             is;
    const PetscInt *indices;
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);            CHKERRQ(ierr);
    ierr = DMPlexGetDataFVM(dm, fvm, NULL, NULL, &dmGrad);          CHKERRQ(ierr);
    ierr = DMCreateDS(dm);                                         CHKERRQ(ierr);
    ierr = DMCreateDS(dmGrad);                                     CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);                                     CHKERRQ(ierr);
    ierr = DMGetDS(dmGrad, &probGrad);                             CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, 0, phys->riemann_solver); CHKERRQ(ierr);
    // TODO: ierr = PetscDSSetRiemannSolver(probGrad, 0, PLUG ADVECTION RIEMANN SOLVER HERE); CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 0, phys);                       CHKERRQ(ierr);
    ierr = PetscDSSetContext(probGrad, 0, phys);                   CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "Face Sets", &label);                    CHKERRQ(ierr);
    { // Getting boundary ids
      IS bnd_is_mpi, bnd_is_loc;
      ierr = DMLabelGetValueIS(label, &bnd_is_loc);                                                    CHKERRQ(ierr);
      ierr = ISOnComm(bnd_is_loc, PetscObjectComm((PetscObject) dm), PETSC_USE_POINTER , &bnd_is_mpi); CHKERRQ(ierr);
      ierr = ISAllGather(bnd_is_mpi, &is);                                                             CHKERRQ(ierr);
      ierr = ISDestroy(&bnd_is_loc);                                                                   CHKERRQ(ierr);
      ierr = ISDestroy(&bnd_is_mpi);                                                                   CHKERRQ(ierr);
      ierr = ISSortRemoveDups(is);                                                                     CHKERRQ(ierr);
      ierr = ISGetSize(is, &phys->nbc);                                                                CHKERRQ(ierr);
      ierr = ISGetIndices(is, &indices);                                                               CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(phys->nbc, &phys->bc_ctx);                 CHKERRQ(ierr);

    PetscFunctionList bcList;
    ierr = BCRegister(&bcList); CHKERRQ(ierr);

    for (PetscInt i = 0; i < phys->nbc; i++) {
      void (*bcFunc)(void);

      phys->bc_ctx[i].phys = phys;
      ierr = YAMLLoadBC(filename, indices[i], phys->dim, phys->bc_ctx + i); CHKERRQ(ierr);
      ierr = PetscFunctionListFind(bcList, phys->bc_ctx[i].type, &bcFunc);  CHKERRQ(ierr);

      if (!bcFunc) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Unknown boundary condition (%s)", phys->bc_ctx[i].type);
      ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, phys->bc_ctx[i].name, label, 1, indices + i, 0, 0, NULL,
                                bcFunc, NULL, phys->bc_ctx + i, NULL); CHKERRQ(ierr);

      ierr = PetscFunctionListFind(bcList, "BC_COPY", &bcFunc);  CHKERRQ(ierr);
      ierr = PetscDSAddBoundary(probGrad, DM_BC_NATURAL_RIEMANN, phys->bc_ctx[i].name, label, 1, indices + i, 0, 0, NULL,
                                bcFunc, NULL, phys->bc_ctx + i, NULL); CHKERRQ(ierr);
    }
    ierr = PetscFunctionListDestroy(&bcList); CHKERRQ(ierr);
    ierr = ISRestoreIndices(is, &indices);    CHKERRQ(ierr);
    ierr = ISDestroy(&is);                    CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);       CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode MeshInsertPeriodicValues(DM dm, Vec locX, Vec locGrad){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  PetscFV  fvm;
  MeshCtx  ctx;
  DM       dmGrad;
  PetscInt Nc, dim;
  { // Getting mesh data
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);   CHKERRQ(ierr);
    ierr = PetscFVGetNumComponents(fvm, &Nc);              CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);                       CHKERRQ(ierr);
    ierr = DMPlexGetDataFVM(dm, fvm, NULL, NULL, &dmGrad); CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, &ctx);              CHKERRQ(ierr);
  }

  PetscReal *locX_array, *locGrad_array, *valX, *valGrad, val[Nc * (dim + 1)];
  ierr = VecGetArray(locX, &locX_array);                     CHKERRQ(ierr);
  if (locGrad) {ierr = VecGetArray(locGrad, &locGrad_array); CHKERRQ(ierr);}
  for (PetscInt n = 0; n < ctx->n_perio; n++) {
    PetscInt       n_master, n_slave;
    const PetscInt *master, *slave;
    PetscReal      *buffer_array;

    ierr = ISGetSize(ctx->perio[n].master, &n_master);  CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->perio[n].master, &master); CHKERRQ(ierr);
    for (PetscInt i = 0; i < n_master; i++) {
      ierr = DMPlexPointLocalFieldRead(dm, master[i], 0, locX_array, &valX);           CHKERRQ(ierr);
      ierr = DMPlexPointLocalFieldRead(dmGrad, master[i], 0, locGrad_array, &valGrad); CHKERRQ(ierr);
      for (PetscInt k = 0; k < Nc; k++) val[k] = valX[k];
      if (locGrad) for (PetscInt k = 0; k < dim * Nc; k++) val[Nc + k] = valGrad[k];
  		ierr = VecSetValuesBlockedLocal(ctx->perio[n].buffer, 1, &i, val, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(ctx->perio[n].master, &master);  CHKERRQ(ierr);
    ierr = VecAssemblyBegin(ctx->perio[n].buffer);           CHKERRQ(ierr);
    ierr = VecAssemblyEnd(ctx->perio[n].buffer);             CHKERRQ(ierr);
    ierr = VecGetArray(ctx->perio[n].buffer, &buffer_array); CHKERRQ(ierr);

    ierr = ISGetSize(ctx->perio[n].slave, &n_slave);  CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->perio[n].slave, &slave); CHKERRQ(ierr);
    for (PetscInt i = 0; i < n_slave; i++) {
      ierr = DMPlexPointLocalFieldRef(dm, slave[i], 0, locX_array, &valX);           CHKERRQ(ierr);
      ierr = DMPlexPointLocalFieldRef(dmGrad, slave[i], 0, locGrad_array, &valGrad); CHKERRQ(ierr);
      for (PetscInt k = 0; k < Nc; k++) valX[k] = buffer_array[Nc * (dim + 1) * i + k];
      if (locGrad) for (PetscInt k = 0; k < dim * Nc; k++) valGrad[k] = buffer_array[Nc * (dim + 1) * i + Nc + k];
    }
    ierr = VecRestoreArray(ctx->perio[n].buffer, &buffer_array); CHKERRQ(ierr);
  }
  if (locGrad) {ierr = VecRestoreArray(locGrad, &locGrad_array); CHKERRQ(ierr);}
  ierr = VecRestoreArray(locX, &locX_array);                     CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MeshReconstructGradientsFVM(DM dm, Vec locX, Vec grad){
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecZeroEntries(grad); CHKERRQ(ierr);

  MeshCtx          ctx;
  Vec              cellgeom;
  DM               dmGrad, dmCell;
  const PetscReal  *x, *cellgeom_a;
  PetscReal        *gr;
  PetscInt         dim, Nc, ncell;
  PetscLimiter     lim;
  PetscLimiterType limType;
  { // Getting mesh data
    PetscFV      fvm;
    ierr = DMGetField(dm, 0, NULL, (PetscObject*) &fvm);        CHKERRQ(ierr);
    ierr = PetscFVGetLimiter(fvm, &lim);                        CHKERRQ(ierr);
    ierr = PetscLimiterGetType(lim, &limType);                  CHKERRQ(ierr);
    if (!strcmp(limType, PETSCLIMITERZERO)) PetscFunctionReturn(0);
    ierr = DMPlexGetDataFVM(dm, fvm, &cellgeom, NULL, &dmGrad); CHKERRQ(ierr);
    ierr = VecGetDM(cellgeom, &dmCell);                         CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom, &cellgeom_a);              CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);                            CHKERRQ(ierr);
    ierr = PetscFVGetNumComponents(fvm, &Nc);                   CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, &ctx);                   CHKERRQ(ierr);
    ierr = VecGetArrayRead(locX, &x);                           CHKERRQ(ierr);
    ierr = VecGetArray(grad, &gr);                              CHKERRQ(ierr);
    ncell = ctx->cStartOverlap - ctx->cStartCell;
  }

  for (PetscInt n = 0; n < ncell; n++) { // Computing cell gradient
    PetscInt       numNeighbors, c = ctx->cStartCell + n;
    const PetscInt *neighborhood;
    PetscReal      *cx, *ncx, *cgrad;

    ierr = ISGetSize(ctx->CellCtx[n].neighborhood, &numNeighbors);    CHKERRQ(ierr);
    ierr = ISGetIndices(ctx->CellCtx[n].neighborhood, &neighborhood); CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, c, x, &cx);                       CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dmGrad, c, gr, &cgrad);               CHKERRQ(ierr);
    for (PetscInt i = 0; i < numNeighbors; i++) {
        ierr = DMPlexPointLocalRead(dm, neighborhood[i], x, &ncx); CHKERRQ(ierr);
        for (PetscInt j = 0; j < Nc; j++) {
          PetscReal delta = ncx[j] - cx[j];
          for (PetscInt k = 0; k < dim; k++) cgrad[j * dim + k] += delta * ctx->CellCtx[n].grad_coeff[dim * i + k];
        }
    }
  }

  for (PetscInt n = (!strcmp(limType, PETSCLIMITERNONE)) ? ncell : 0; n < ncell; n++) { // Limit cell gradient
    const PetscInt  *faces, *fcells;
    PetscInt        nface, c = ctx->cStartCell + n, nc;
    PetscReal       *cx, *ncx, *cgrad;
    PetscFVCellGeom *cg, *ncg;
    PetscReal cellPhi[Nc];
    ierr = DMPlexGetConeSize(dm, c, &nface);                 CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, c, &faces);                     CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm, c, x, &cx);              CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dmGrad, c, gr, &cgrad);      CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, c, cellgeom_a, &cg); CHKERRQ(ierr);
    for (PetscInt i = 0; i < Nc; i++) cellPhi[i] = PETSC_MAX_REAL;
    for (PetscInt f = 0; f < nface; f++) {
      PetscReal       dx[dim];

      ierr = DMPlexGetSupport(dm, faces[f], &fcells);            CHKERRQ(ierr);
      nc = (c == fcells[0]) ? fcells[1] : fcells[0];
      ierr = DMPlexPointLocalRead(dm, nc, x, &ncx);              CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, nc, cellgeom_a, &ncg); CHKERRQ(ierr);
      for (PetscInt j = 0; j < dim; j++) dx[j] = ncg->centroid[j] - cg->centroid[j];
      for (PetscInt i = 0; i < Nc; i++) {
        PetscReal denom = 0, phi;
        for (PetscInt j = 0; j < dim; j++) denom += cgrad[i * dim + j] * dx[j];
        ierr = PetscLimiterLimit(lim, (ncx[i] - cx[i]) / (2 * denom), &phi); CHKERRQ(ierr);
        cellPhi[i] = PetscMin(cellPhi[i], phi);
      }
    }
    for (PetscInt i = 0; i < Nc; i++) {
      for (PetscInt j = 0; j < dim; j++) cgrad[i * dim + j] *= cellPhi[i];
    }
  }

  { // Cleanup
    ierr = VecRestoreArrayRead(cellgeom, &cellgeom_a); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locX, &x);              CHKERRQ(ierr);
    ierr = VecRestoreArray(grad, &gr);                 CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode VecApplyFunctionInPlace(Vec x, void (*func)(const PetscReal*, PetscReal*, void*), void *ctx){
  PetscErrorCode ierr;
  PetscInt       Nc, size;
  PetscReal      *val_x;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(x, &size);  CHKERRQ(ierr);
  ierr = VecGetBlockSize(x, &Nc);    CHKERRQ(ierr);
  ierr = VecGetArray(x, &val_x);     CHKERRQ(ierr);
  for (PetscInt i = 0; i < size; i += Nc) func(val_x + i, val_x + i, ctx);
  ierr = VecRestoreArray(x, &val_x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecApplyFunctionComponents(Vec x, Vec *y,
                                          void (*func)(const PetscReal*, PetscReal*, void*),
                                          void *ctx){
  PetscErrorCode ierr;
  PetscInt       Nc, start, end, size;

  PetscFunctionBeginUser;
  ierr = VecGetBlockSize(x, &Nc);    CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x, &start, &end);        CHKERRQ(ierr);
  size = (end - start) / Nc;

  ierr = VecCreate(PetscObjectComm((PetscObject) x), y); CHKERRQ(ierr);
  ierr = VecSetSizes(*y, size, PETSC_DECIDE);            CHKERRQ(ierr);
  ierr = VecSetFromOptions(*y);                          CHKERRQ(ierr);

  const PetscReal *val_x;
  PetscReal       *val_y;
  PetscInt        *ix;

  ierr = PetscMalloc2(size, &val_y, size, &ix);  CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &val_x);             CHKERRQ(ierr);
  for (PetscInt i = 0; i < size; i++) {
    func(&val_x[Nc * i], &val_y[i], ctx);
    ix[i] = i + start / Nc;
  }
  ierr = VecRestoreArrayRead(x, &val_x);                   CHKERRQ(ierr);
  ierr = VecSetValues(*y, size, ix, val_y, INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(*y);                             CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*y);                               CHKERRQ(ierr);
  ierr = PetscFree2(val_y, ix);                            CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
