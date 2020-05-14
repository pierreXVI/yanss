#ifndef INCLUDE_FLAG_INPUT
#define INCLUDE_FLAG_INPUT

#include "utils.h"

/*
  Functions for input file parsing
  The input is a yaml file, and follows the structure :

  Physics:
    gamma: value

  InitialConditions:
    field1D: value
    fieldND: array
    ...

  BoundaryConditions:
    id:
      name: value
      type: value
      [field1D: value]
      [fieldND: array]
      ...

  [PETScOptions:]
    [- ...]

*/

/*
  Look for the key `varname` in the file named `filename`
  The location of the key is specified in the array `loc`, of size `depth` :
    loc[0] > loc[1] > ... > loc[depth - 1] > varname : var
*/
PetscErrorCode IOSeekVarFromLoc(const char*, const char*, PetscInt, const char**, PetscBool*);

/*
  Load the value associated to the key `varname` in the file named `filename`
  The location of the key is specified in the array `loc`, of size `depth` :
    loc[0] > loc[1] > ... > loc[depth - 1] > varname : var
    The output value must be freed with `PetscFree`
*/
PetscErrorCode IOLoadVarFromLoc(const char*, const char*, PetscInt, const char**, const char**);

/*
  Load the array of values associated to the key `varname` in the file named `filename`
  The location of the key is specified in the array `loc`, of size `depth` :
    loc[0] > loc[1] > ... > loc[depth - 1] > varname : [array of length `len`]
  If the pointer `PetscInt *len` is set to a > 0 size, read the asked size, and if it is set to 0, it is set to the read array size
  The output value must be freed with `PetscFree`
*/
PetscErrorCode IOLoadVarArrayFromLoc(const char*, const char*, PetscInt, const char**, PetscInt*, const char***);


/*
  Load the boundary condition with the right id from the input file
  The boundary condition `name` and `val` are allocated and must be freed with `PetscFree`
*/
PetscErrorCode IOLoadBC(const char*, const PetscInt, PetscInt, struct BCDescription*);

/*
  Load the initial condition from the input file
  The output array is allocated and must be freed with `PetscFree`
*/
PetscErrorCode IOLoadInitialCondition(const char*, PetscInt, PetscReal**);

/*
  Load the list of string options and register them into the database
*/
PetscErrorCode IOLoadPetscOptions(const char*);

#endif
