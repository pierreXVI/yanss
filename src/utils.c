#include "utils.h"


PetscErrorCode MyStrdup(const char *in, const char **out){
  PetscErrorCode ierr;
  size_t         len;
  char           *new;

  PetscFunctionBeginUser;
  ierr = PetscStrlen(in, &len);       CHKERRQ(ierr);
  ierr = PetscMalloc1(len + 1, &new); CHKERRQ(ierr);
  ierr = PetscStrcpy(new, in);        CHKERRQ(ierr);
  *out = new;
  PetscFunctionReturn(0);
}
