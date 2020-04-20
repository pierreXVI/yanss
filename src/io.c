#include "io.h"
#include <yaml.h>

#define ERR_HIGHLIGHT "\e[1;33m"

/*
  Linked list of `yaml_event_t`
*/
struct yaml_event_list {
  yaml_event_t           event;
  struct yaml_event_list *next;
};

/*
  Delete an event linked list, fron the node to the end
*/
static PetscErrorCode yaml_event_list_delete(struct yaml_event_list *node){
  PetscErrorCode ierr;
  struct yaml_event_list *next = node;

  PetscFunctionBeginUser;
  while(next) {
    node = next;
    yaml_event_delete(&node->event);
    next = node->next;
    ierr = PetscFree(node); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*
  Create a parser with the given input file.
  The parser must be destroyed with `yaml_parser_my_delete`
*/
static PetscErrorCode yaml_parser_my_initialize(yaml_parser_t *parser, const char *filename) {
  PetscFunctionBeginUser;
  FILE *input = fopen(filename, "rb");
  if (!input) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "%sFailed to open %s\e[0;39m", ERR_HIGHLIGHT, filename);
  }
  if (!yaml_parser_initialize(parser)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER, "%sCannot initialize parser\e[0;39m", ERR_HIGHLIGHT);
  yaml_parser_set_input_file(parser, input);
  PetscFunctionReturn(0);
}

/*
  Delete the given parser and close the input file
*/
static PetscErrorCode yaml_parser_my_delete(yaml_parser_t *parser) {
  PetscFunctionBeginUser;
  FILE *input = parser->input.file;
  yaml_parser_delete(parser);
  if(fclose(input)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SYS, "%sError on fclose\e[0;39m", ERR_HIGHLIGHT);
  PetscFunctionReturn(0);
}

/*
  Get the next event from the parser
*/
static PetscErrorCode yaml_parser_my_parse(yaml_parser_t *parser, yaml_event_t *event) {
  PetscFunctionBeginUser;
  if (!yaml_parser_parse(parser, event)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "%sFailed to parse: %s\e[0;39m", ERR_HIGHLIGHT, parser->problem);
  PetscFunctionReturn(0);
}


/*
  Seek for the scalar 'scalarname' at the current level
*/
static PetscErrorCode IOMoveToScalar(yaml_parser_t *parser, const char *filename, const char *scalarname){
  PetscErrorCode ierr;
  PetscInt       level = -2;

  PetscFunctionBeginUser;

  while (PETSC_TRUE) {
    yaml_event_t event;
    ierr = yaml_parser_my_parse(parser, &event); CHKERRQ(ierr);
    yaml_event_type_t event_type = event.type;

    level = (level == -2) ? -1 : level;

    if (level == 0 && event_type == YAML_SCALAR_EVENT && !strcmp(scalarname, (const char*) event.data.scalar.value)) {
      yaml_event_delete(&event);
      PetscFunctionReturn(0);
    }
    yaml_event_delete(&event);
    switch (event_type) {
    case YAML_STREAM_START_EVENT:   continue; break;
    case YAML_DOCUMENT_START_EVENT: continue; break;
    case YAML_MAPPING_START_EVENT:  level++; break;
    case YAML_MAPPING_END_EVENT:    level--; break;
    default: break;
    }
    if (level < 0){
      SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "%s%s not found in %s\e[0;39m", ERR_HIGHLIGHT, scalarname, filename);
    }
  }
}


#define CHKERRQ_PARSER(ierr, parser)  if ((ierr)) {CHKERRQ(yaml_parser_my_delete(&(parser)));  CHKERRQ(ierr);}

PetscErrorCode IOLoadVarFromLoc(const char *filename, const char *varname, PetscInt depth, const char **loc, const char **var){
  PetscErrorCode ierr;
  yaml_parser_t  parser;

  PetscFunctionBeginUser;
  ierr = yaml_parser_my_initialize(&parser, filename);         CHKERRQ(ierr);

  while (depth > 0) {
    ierr = IOMoveToScalar(&parser, filename, loc[0]);          CHKERRQ_PARSER(ierr, parser);
    depth--;
    loc++;
  }
  ierr = IOMoveToScalar(&parser, filename, varname);           CHKERRQ_PARSER(ierr, parser);

  yaml_event_t event;
  ierr = yaml_parser_my_parse(&parser, &event);                CHKERRQ_PARSER(ierr, parser);
  if (event.type != YAML_SCALAR_EVENT) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "%sCannot read %s in %s\e[0;39m", ERR_HIGHLIGHT, varname, filename);
  ierr = MyStrdup((const char*) event.data.scalar.value, var); CHKERRQ_PARSER(ierr, parser);
  yaml_event_delete(&event);

  ierr = yaml_parser_my_delete(&parser);                       CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IOLoadVarArrayFromLoc(const char *filename, const char *varname, PetscInt depth, const char **loc, PetscInt *len, const char ***var){
  PetscErrorCode ierr;
  yaml_parser_t  parser;

  PetscFunctionBeginUser;
  ierr = yaml_parser_my_initialize(&parser, filename); CHKERRQ(ierr);

  while (depth > 0) {
    ierr = IOMoveToScalar(&parser, filename, loc[0]);  CHKERRQ_PARSER(ierr, parser);
    depth--;
    loc++;
  }
  ierr = IOMoveToScalar(&parser, filename, varname);   CHKERRQ_PARSER(ierr, parser);

  struct yaml_event_list *root = PETSC_NULL, *current, *node;
  PetscInt               done = -1;
  *len = 0;
  while (PETSC_TRUE) {
    ierr = PetscNew(&node);                             CHKERRQ_PARSER(ierr, parser);
    ierr = yaml_parser_my_parse(&parser, &node->event); CHKERRQ_PARSER(ierr, parser);
    node->next = PETSC_NULL;

    switch (node->event.type) {
    case YAML_SEQUENCE_START_EVENT:
      done = (done == -1) ? 0 : -1;
      ierr = yaml_event_list_delete(node);             CHKERRQ_PARSER(ierr, parser);
      break;
    case YAML_SEQUENCE_END_EVENT:
      done = (done ==  0) ? 1 : -1;
      ierr = yaml_event_list_delete(node);             CHKERRQ_PARSER(ierr, parser);
      break;
    case YAML_SCALAR_EVENT:
      (*len)++;
      if (!root) {
        current = root = node;
      } else {
        current = current->next = node;
      }
      break;
    default:
      done = -1;
      ierr = yaml_event_list_delete(node);             CHKERRQ_PARSER(ierr, parser);
      break;
    }

    if (done == 1) {break;}
    else if (done == -1) {
      ierr = yaml_event_list_delete(root);             CHKERRQ_PARSER(ierr, parser);
      ierr = yaml_parser_my_delete(&parser);           CHKERRQ(ierr);
      SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "%sCannot read list %s in %s\e[0;39m", ERR_HIGHLIGHT, varname, filename);
    }
  }

  ierr = PetscMalloc1(*len, var); CHKERRQ_PARSER(ierr, parser);

  PetscInt i = 0;
  for (node = root; node; node = node->next){
    ierr = MyStrdup((const char*) node->event.data.scalar.value, (*var) + i++); CHKERRQ_PARSER(ierr, parser);
  }
  ierr = yaml_event_list_delete(root);                                          CHKERRQ_PARSER(ierr, parser);

  ierr = yaml_parser_my_delete(&parser); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#define CHKERRQ_BC(ierr, bc) do {if (PetscUnlikely(ierr)) {PetscErrorPrintf("%sCannot read boundaryConditions > %s\e[0;39m\n", ERR_HIGHLIGHT, (bc)); CHKERRQ(ierr);}} while(0)
PetscErrorCode IOLoadBC(const char *filename, const PetscInt id, const PetscInt dim, struct BCDescription *bc){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  char id_str[8];
  ierr = PetscSNPrintf(id_str, sizeof(id_str), "%d", id);

  const char* loc[] = {"boundaryConditions", id_str};

  ierr = IOLoadVarFromLoc(filename, "name", 2, loc, &bc->name); CHKERRQ_BC(ierr, id_str);

  const char *buffer_type;
  ierr = IOLoadVarFromLoc(filename, "type", 2, loc, &buffer_type); CHKERRQ_BC(ierr, id_str);
  if (!strcmp(buffer_type, "BC_DIRICHLET")) {
    bc->type = BC_DIRICHLET;
    ierr = PetscMalloc1(dim + 2, &bc->val); CHKERRQ(ierr);

    const char *buffer_val;
    const char **buffer_vals;
    PetscInt   size;

    ierr = IOLoadVarFromLoc(filename, "rho", 2, loc, &buffer_val); CHKERRQ_BC(ierr, id_str);
    bc->val[0] = atof(buffer_val);
    ierr = PetscFree(buffer_val);                                  CHKERRQ(ierr);

    ierr = IOLoadVarArrayFromLoc(filename, "u", 2, loc, &size, &buffer_vals); CHKERRQ_BC(ierr, id_str);
    if (size < dim) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "%sFailed to read boundaryConditions > %s > u : not enough values (expected %d, got %d)\e[0;39m", ERR_HIGHLIGHT, id_str, dim, size);
    for (PetscInt i = 0; i < dim; i++) {bc->val[1 + i] = atof(buffer_vals[i]);}
    for (PetscInt i = 0; i < size; i++) {ierr = PetscFree(buffer_vals[i]);    CHKERRQ(ierr);}
    ierr = PetscFree(buffer_vals);                                            CHKERRQ(ierr);

    ierr = IOLoadVarFromLoc(filename, "p", 2, loc, &buffer_val); CHKERRQ_BC(ierr, id_str);
    bc->val[dim + 1] = atof(buffer_val);
    ierr = PetscFree(buffer_val);                                CHKERRQ(ierr);

  } else if (!strcmp(buffer_type, "BC_OUTFLOW_P")) {
    bc->type = BC_OUTFLOW_P;
    ierr = PetscMalloc1(1, &bc->val); CHKERRQ(ierr);

    const char *buffer_val;
    ierr = IOLoadVarFromLoc(filename, "p", 2, loc, &buffer_val); CHKERRQ_BC(ierr, id_str);
    bc->val[0] = atof(buffer_val);
    ierr = PetscFree(buffer_val);                                CHKERRQ(ierr);

  } else if (!strcmp(buffer_type, "BC_WALL")) {
    bc->type = BC_WALL;
    bc->val = PETSC_NULL;
  }
  else {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "%sUnknown boundary condition (%s)\n", ERR_HIGHLIGHT, buffer_type);
  }
  ierr = PetscFree(buffer_type);                                   CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#define CHKERRQ_IC(ierr) do {if (PetscUnlikely(ierr)) {PetscErrorPrintf("%sCannot read initialConditions\e[0;39m\n", ERR_HIGHLIGHT); CHKERRQ(ierr);}} while(0)
PetscErrorCode IOLoadInitialCondition(const char *filename, const PetscInt dim, PetscReal **initialConditions){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscMalloc1(dim + 2, initialConditions); CHKERRQ(ierr);

  const char *loc = "initialConditions";

  const char *buffer_val;
  const char **buffer_vals;
  PetscInt   size;

  ierr = IOLoadVarFromLoc(filename, "rho", 1, &loc, &buffer_val); CHKERRQ_IC(ierr);
  (*initialConditions)[0] = atof(buffer_val);
  ierr = PetscFree(buffer_val);                                   CHKERRQ(ierr);

  ierr = IOLoadVarArrayFromLoc(filename, "u", 1, &loc, &size, &buffer_vals); CHKERRQ_IC(ierr);
  if (size < dim) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "%sFailed to read initialConditions > u : not enough values (expected %d, got %d)\e[0;39m", ERR_HIGHLIGHT, dim, size);
  for (PetscInt i = 0; i < dim; i++) {(*initialConditions)[1 + i] = atof(buffer_vals[i]);}
  for (PetscInt i = 0; i < size; i++) {ierr = PetscFree(buffer_vals[i]);     CHKERRQ(ierr);}
  ierr = PetscFree(buffer_vals);                                             CHKERRQ(ierr);

  ierr = IOLoadVarFromLoc(filename, "p", 1, &loc, &buffer_val); CHKERRQ_IC(ierr);
  (*initialConditions)[dim + 1] = atof(buffer_val);
  ierr = PetscFree(buffer_val);                                 CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
