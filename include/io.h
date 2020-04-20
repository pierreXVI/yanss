#ifndef IO
#define IO

#include "utils.h"
#include <yaml.h>

/*
  Load the value associated to the key `varname` in the file named `filename`
  The location of the key is specified in the array `loc`, of size `depth` :
    loc[0] > loc[1] > ... > loc[depth - 1] > varname : var
*/
PetscErrorCode IOLoadVarFromLoc(const char*, const char*, PetscInt, const char**, const char**);

/*
  Load the array of values associated to the key `varname` in the file named `filename`
  The location of the key is specified in the array `loc`, of size `depth` :
    loc[0] > loc[1] > ... > loc[depth - 1] > varname : [array of length `len`]
*/
PetscErrorCode IOLoadVarArrayFromLoc(const char*, const char*, PetscInt, const char**, PetscInt*, const char***);


PetscErrorCode IOLoadBC(const char*, const PetscInt, const PetscInt, struct BCDescription*);

PetscErrorCode IOLoadInitialCondition(const char*, const PetscInt, PetscReal**);


#endif
