#ifndef INCLUDE_FLAG_OUTPUT
#define INCLUDE_FLAG_OUTPUT

#include "utils.h"


struct Monitor_ctx {
  PetscInt    n_iter; // Monitor evry `n_iter` iteration
  PetscViewer viewer; // PetscViewer used by the monitor
};


/*
  TS monitors, with the calling sequence:

  ```
  PetscErrorCode monitor(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx)
    ts    - TS context
    steps - Iteration number
    time  - Current time
    u     - Current iterate
    mctx  - Monitoring context, to be casted to (struct Monitor_ctx*)
  ```
*/

/*
  Print for each field "field_name : min_value, max_value"
*/
PetscErrorCode IOMonitorAscii_MinMax(TS, PetscInt, PetscReal, Vec, void*);

/*
  Print for each field "field_name : min_flux_value, max_fluxvalue"
*/
PetscErrorCode IOMonitorAscii_Res(TS, PetscInt, PetscReal, Vec, void*);

/*
  Draw each fields in separate graphic windows
*/
PetscErrorCode IOMonitorDraw(TS, PetscInt, PetscReal, Vec, void*);

/*
  Draw norm of the velocity
*/
PetscErrorCode IOMonitorDrawNormU(TS, PetscInt, PetscReal, Vec, void*);

PetscErrorCode IOMonitorDEBUG(TS, PetscInt, PetscReal, Vec, void*);

#endif
