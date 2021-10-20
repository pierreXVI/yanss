#ifndef INCLUDE_FLAG_OUTPUT
#define INCLUDE_FLAG_OUTPUT

#include "utils.h"


struct MonitorCtx {
  PetscInt    n_iter; // Monitor every `n_iter` iteration
  PetscViewer viewer; // PetscViewer used by the monitor
  Physics     phys;   // Physical model
};


/*
  TS monitors, with the calling sequence:

  ```
  PetscErrorCode monitor(TS ts, PetscInt steps, PetscReal time, Vec u, void *ctx)
    ts    - TS context
    steps - Iteration number
    time  - Current time
    u     - Current iterate
    ctx   - Monitoring context, to be cast to (struct Monitor_ctx*)
  ```
*/

/*
  Print for each field "field_name : min_value, max_value"
*/
PetscErrorCode MonitorAscii_MinMax(TS, PetscInt, PetscReal, Vec, void*);

/*
  Print for each field "field_name : min_flux_value, max_flux_value"
*/
PetscErrorCode MonitorAscii_Res(TS, PetscInt, PetscReal, Vec, void*);

/*
  Draw each fields in separate graphic windows
*/
PetscErrorCode MonitorDraw(TS, PetscInt, PetscReal, Vec, void*);

/*
  Draw norm of the velocity
*/
PetscErrorCode MonitorDrawNormU(TS, PetscInt, PetscReal, Vec, void*);

/*
  Draw each field gradients in separate graphic windows
*/
PetscErrorCode MonitorDrawGrad(TS, PetscInt, PetscReal, Vec, void*);

PetscErrorCode MonitorDEBUG(TS, PetscInt, PetscReal, Vec, void*);

#endif
