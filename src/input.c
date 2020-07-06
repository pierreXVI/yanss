#include "input.h"
#include <yaml.h>

/*
  Higligter for parser errors
*/
#define PARSER_ERROR_HIGHLIGHT "\e[1;33m"

static char ERR_HEADER[256] = "";

typedef yaml_parser_t *IOParser;


/*
  Destroy the given IOParser and close the input file
*/
static PetscErrorCode IOParserDestroy(IOParser *parser) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  FILE *input = (*parser)->input.file;
  yaml_parser_delete(*parser);
  ierr = PetscFree(*parser); CHKERRQ(ierr);
  *parser = PETSC_NULL;
  if (fclose(input)) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SYS, "Error on fclose");
  ierr = PetscPopErrorHandler(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Error handler wrapper for parsing errors:
    Replace `mess` with : `PARSER_ERROR_HIGHLIGHT` + `ERR_HEADER` + `mess` + "\e[0;39m"
    Delete the IOParser
*/
static PetscErrorCode ParserErrorHandler(MPI_Comm comm, int line, const char *func, const char *file, PetscErrorCode n, PetscErrorType p, const char *mess, void *ctx){
  size_t         len, n1, n2, n3, n4;
  char           *error_msg;
  IOParser       *parser = (IOParser*) ctx;

  PetscFunctionBeginUser;
  PetscStrlen(PARSER_ERROR_HIGHLIGHT, &n1);
  PetscStrlen(ERR_HEADER, &n2);
  PetscStrlen(mess, &n3);
  PetscStrlen("\e[0;39m", &n4);
  PetscMalloc1(n1 + n2 + n3 + n4 + 1, &error_msg);
  PetscSNPrintf(error_msg, sizeof(error_msg), "%s%s%s\e[0;39m", PARSER_ERROR_HIGHLIGHT, ERR_HEADER, mess);
  PetscTraceBackErrorHandler(comm, line, func, file, n, p, error_msg, ctx);
  PetscFree(error_msg);
  if (parser) IOParserDestroy(parser);
  PetscFunctionReturn(n);
}

/*
  Create a parser with the given input file.
  The parser must be destroyed with `IOParserDestroy`
*/
static PetscErrorCode IOParserCreate(IOParser *parser, const char *filename) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  parser = PETSC_NULL;
  ierr = PetscPushErrorHandler(ParserErrorHandler, parser); CHKERRQ(ierr);

  FILE *input = fopen(filename, "rb");
  if (!input) {
    SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "%sFailed to open %s\e[0;39m", PARSER_ERROR_HIGHLIGHT, filename);
  }
  ierr = PetscNew(parser); CHKERRQ(ierr);

  if (!yaml_parser_initialize(*parser)) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "%sCannot initialize parser\e[0;39m", PARSER_ERROR_HIGHLIGHT);
  yaml_parser_set_input_file(*parser, input);
  PetscFunctionReturn(0);
}

/*
  Get the next event from the IOParser
*/
static PetscErrorCode IOParserParse(IOParser parser, yaml_event_t *event) {
  PetscFunctionBeginUser;
  if (!yaml_parser_parse(parser, event)) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_FILE_UNEXPECTED, "Failed to parse: %s", parser->problem);
  PetscFunctionReturn(0);
}


/*
  Seek for the anchor `anchorname` from the current parser position to the end
  If event_p is not PETSC_NULL, parses and returns the next event
*/
static PetscErrorCode IOMoveToAnchor(IOParser parser, const char *filename, const char *anchorname, yaml_event_t *event_p){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  while (PETSC_TRUE) {
    yaml_event_t event;
    ierr = IOParserParse(parser, &event); CHKERRQ(ierr);
    yaml_event_type_t event_type = event.type;

    if (event_type == YAML_SCALAR_EVENT && event.data.scalar.anchor && !strcmp(anchorname, (const char*) event.data.scalar.anchor)) {
      yaml_event_delete(&event);
      if (event_p) {
        ierr = IOParserParse(parser, event_p); CHKERRQ(ierr);
        if (event_p->type == YAML_ALIAS_EVENT) {
          const char *newanchorname;
          ierr = MyStrdup(event_p->data.alias.anchor, &newanchorname);     CHKERRQ(ierr);
          yaml_event_delete(event_p);
          ierr = IOParserDestroy(&parser);                                 CHKERRQ(ierr);
          ierr = IOParserCreate(&parser, filename);                        CHKERRQ(ierr);
          ierr = IOMoveToAnchor(parser, filename, newanchorname, event_p); CHKERRQ(ierr);
          ierr = PetscFree(newanchorname);                                 CHKERRQ(ierr);
        }
      }
      PetscFunctionReturn(0);
    }

    yaml_event_delete(&event);
    if (event_type == YAML_DOCUMENT_END_EVENT) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "&%s not found in %s", anchorname, filename);
  }
}

/*
  Seek for the scalar `scalarname` at the current level
  If event_p is not PETSC_NULL, parses and returns the next event
*/
static PetscErrorCode IOMoveToScalar(IOParser parser, const char *filename, const char *scalarname, yaml_event_t *event_p){
  PetscErrorCode ierr;
  PetscInt       level = -1;

  PetscFunctionBeginUser;

  while (PETSC_TRUE) {
    yaml_event_t event;
    ierr = IOParserParse(parser, &event); CHKERRQ(ierr);
    yaml_event_type_t event_type = event.type;

    if (event_type == YAML_SCALAR_EVENT && level == 0 && !strcmp(scalarname, (const char*) event.data.scalar.value)) {
      yaml_event_delete(&event);
      if (event_p) {
        ierr = IOParserParse(parser, event_p); CHKERRQ(ierr);
        if (event_p->type == YAML_ALIAS_EVENT) {
          const char *anchorname;
          ierr = MyStrdup(event_p->data.alias.anchor, &anchorname);     CHKERRQ(ierr);
          yaml_event_delete(event_p);
          ierr = IOParserDestroy(&parser);                              CHKERRQ(ierr);
          ierr = IOParserCreate(&parser, filename);                     CHKERRQ(ierr);
          ierr = IOMoveToAnchor(parser, filename, anchorname, event_p); CHKERRQ(ierr);
          ierr = PetscFree(anchorname);                                 CHKERRQ(ierr);
        }
      }
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
      SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "%s not found in %s", scalarname, filename);
    }
  }
}


PetscErrorCode IOSeekVarFromLoc(const char *filename, const char *varname, PetscInt depth, const char **loc, PetscBool *found){
  PetscErrorCode ierr;
  IOParser       parser;

  PetscFunctionBeginUser;
  ierr = IOParserCreate(&parser, filename);                      CHKERRQ(ierr);
  while (depth > 0) {
    ierr = IOMoveToScalar(parser, filename, loc[0], PETSC_NULL); CHKERRQ(ierr);
    depth--;
    loc++;
  }
  ierr = PetscPushErrorHandler(PetscReturnErrorHandler, PETSC_NULL); CHKERRQ(ierr);
  ierr = IOMoveToScalar(parser, filename, varname, PETSC_NULL);
  *found = (ierr) ? PETSC_FALSE : PETSC_TRUE;
  ierr = PetscPopErrorHandler();   CHKERRQ(ierr);
  ierr = IOParserDestroy(&parser); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IOLoadVarFromLoc(const char *filename, const char *varname, PetscInt depth, const char **loc, const char **var){
  PetscErrorCode ierr;
  IOParser       parser;
  yaml_event_t   event;

  PetscFunctionBeginUser;
  ierr = IOParserCreate(&parser, filename); CHKERRQ(ierr);

  while (depth > 0) {
    ierr = IOMoveToScalar(parser, filename, loc[0], PETSC_NULL); CHKERRQ(ierr);
    depth--;
    loc++;
  }
  ierr = IOMoveToScalar(parser, filename, varname, &event);      CHKERRQ(ierr);
  if (event.type != YAML_SCALAR_EVENT) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Cannot read %s in %s", varname, filename);
  ierr = MyStrdup((const char*) event.data.scalar.value, var);   CHKERRQ(ierr);
  yaml_event_delete(&event);

  ierr = IOParserDestroy(&parser); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IOLoadVarArrayFromLoc(const char *filename, const char *varname, PetscInt depth, const char **loc, PetscInt *len, const char ***var){
  PetscErrorCode ierr;
  IOParser       parser;
  yaml_event_t   event;

  PetscFunctionBeginUser;
  ierr = IOParserCreate(&parser, filename); CHKERRQ(ierr);

  while (depth > 0) {
    ierr = IOMoveToScalar(parser, filename, loc[0], PETSC_NULL); CHKERRQ(ierr);
    depth--;
    loc++;
  }
  ierr = IOMoveToScalar(parser, filename, varname, &event);      CHKERRQ(ierr);

  struct yaml_event_list {yaml_event_t event; struct yaml_event_list *next;} *root = PETSC_NULL, *current, *node;
  PetscInt done = -1, i = 0;
  while (PETSC_TRUE) {
    ierr = PetscNew(&node); CHKERRQ(ierr);
    if (done == -1) {
      node->event = event;
    } else {
      ierr = IOParserParse(parser, &node->event); CHKERRQ(ierr);
    }
    node->next = PETSC_NULL;

    switch (node->event.type) {
    case YAML_SCALAR_EVENT:
      if (done == -1) {
        i = 0;
        done = 1;
        break;
      }
      i++;
      if (!root) {
        current = root = node;
      } else {
        current = current->next = node;
      }
      break;
    case YAML_SEQUENCE_START_EVENT:
      done = (done == -1) ? 0 : -1;
      yaml_event_delete(&node->event);
      ierr = PetscFree(node); CHKERRQ(ierr);
      break;
    case YAML_SEQUENCE_END_EVENT:
      done = (done ==  0) ? 1 : -1;
      yaml_event_delete(&node->event);
      ierr = PetscFree(node); CHKERRQ(ierr);
      break;
    default:
      done = -1;
      yaml_event_delete(&node->event);
      ierr = PetscFree(node); CHKERRQ(ierr);
      break;
    }

    if (done == 1) {
      break;
    } else if (done == -1) {
      node = root;
      while (node) {
        root = node;
        yaml_event_delete(&root->event);
        node = root->next;
        ierr = PetscFree(root); CHKERRQ(ierr);
      }
      SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Cannot read list %s in %s", varname, filename);
    }
  }

  if (*len > 0 && i < *len) {
    SETERRQ4(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Cannot read list %s in %s: not enough values (expected %d, got %d)", varname, filename, *len, i);
  } else if (*len == 0) {
    *len = i;
  }

  ierr = PetscMalloc1(*len, var); CHKERRQ(ierr);

  node = root;
  for (PetscInt i = 0; i < *len; i++){
    ierr = MyStrdup((const char*) node->event.data.scalar.value, (*var) + i); CHKERRQ(ierr);
    node = node->next;
  }

  node = root;
  while (node) {
    root = node;
    yaml_event_delete(&root->event);
    node = root->next;
    ierr = PetscFree(root); CHKERRQ(ierr);
  }
  ierr = IOParserDestroy(&parser); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



PetscErrorCode IOLoadBC(const char *filename, const PetscInt id, PetscInt dim, struct BCDescription *bc){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscSNPrintf(ERR_HEADER, sizeof(ERR_HEADER), "Cannot read BoundaryConditions > %d: ", id); CHKERRQ(ierr);

  char id_str[8];
  ierr = PetscSNPrintf(id_str, sizeof(id_str), "%d", id);

  const char* loc[] = {"BoundaryConditions", id_str};

  ierr = IOLoadVarFromLoc(filename, "name", 2, loc, &bc->name); CHKERRQ(ierr);

  const char *buffer_type;
  ierr = IOLoadVarFromLoc(filename, "type", 2, loc, &buffer_type); CHKERRQ(ierr);
  if (!strcmp(buffer_type, "BC_DIRICHLET")) {
    bc->type = BC_DIRICHLET;
    ierr = PetscMalloc1(dim + 2, &bc->val); CHKERRQ(ierr);

    const char *buffer_val;
    const char **buffer_vals;

    ierr = IOLoadVarFromLoc(filename, "rho", 2, loc, &buffer_val); CHKERRQ(ierr);
    bc->val[0] = atof(buffer_val);
    ierr = PetscFree(buffer_val);                                  CHKERRQ(ierr);

    ierr = IOLoadVarArrayFromLoc(filename, "u", 2, loc, &dim, &buffer_vals); CHKERRQ(ierr);
    for (PetscInt i = 0; i < dim; i++) {
      bc->val[1 + i] = atof(buffer_vals[i]);
      ierr = PetscFree(buffer_vals[i]);                                      CHKERRQ(ierr);
    }
    ierr = PetscFree(buffer_vals);                                           CHKERRQ(ierr);

    ierr = IOLoadVarFromLoc(filename, "p", 2, loc, &buffer_val); CHKERRQ(ierr);
    bc->val[dim + 1] = atof(buffer_val);
    ierr = PetscFree(buffer_val);                                CHKERRQ(ierr);

  } else if (!strcmp(buffer_type, "BC_OUTFLOW_P")) {
    bc->type = BC_OUTFLOW_P;
    ierr = PetscMalloc1(1, &bc->val); CHKERRQ(ierr);

    const char *buffer_val;
    ierr = IOLoadVarFromLoc(filename, "p", 2, loc, &buffer_val); CHKERRQ(ierr);
    bc->val[0] = atof(buffer_val);
    ierr = PetscFree(buffer_val);                                CHKERRQ(ierr);

  } else if (!strcmp(buffer_type, "BC_WALL")) {
    bc->type = BC_WALL;
    bc->val = PETSC_NULL;
  } else if (!strcmp(buffer_type, "BC_FARFIELD")) {
    bc->type = BC_FARFIELD;
    bc->val = PETSC_NULL;
  } else {
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Unknown boundary condition (%s)", buffer_type);
  }
  ierr = PetscFree(buffer_type); CHKERRQ(ierr);

  ERR_HEADER[0] = '\0';
  PetscFunctionReturn(0);
}


PetscErrorCode IOLoadInitialCondition(const char *filename, PetscInt dim, PetscReal **initialConditions){
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscSNPrintf(ERR_HEADER, sizeof(ERR_HEADER), "Cannot read InitialConditions: "); CHKERRQ(ierr);
  ierr = PetscMalloc1(dim + 2, initialConditions);                                         CHKERRQ(ierr);

  const char *loc = "InitialConditions";
  const char *buffer_val;
  const char **buffer_vals;

  ierr = IOLoadVarFromLoc(filename, "rho", 1, &loc, &buffer_val); CHKERRQ(ierr);
  (*initialConditions)[0] = atof(buffer_val);
  ierr = PetscFree(buffer_val);                                   CHKERRQ(ierr);

  ierr = IOLoadVarArrayFromLoc(filename, "u", 1, &loc, &dim, &buffer_vals); CHKERRQ(ierr);
  for (PetscInt i = 0; i < dim; i++) {
    (*initialConditions)[1 + i] = atof(buffer_vals[i]);
    ierr = PetscFree(buffer_vals[i]);                                       CHKERRQ(ierr);
  }
  ierr = PetscFree(buffer_vals);                                            CHKERRQ(ierr);

  ierr = IOLoadVarFromLoc(filename, "p", 1, &loc, &buffer_val); CHKERRQ(ierr);
  (*initialConditions)[dim + 1] = atof(buffer_val);
  ierr = PetscFree(buffer_val);                                 CHKERRQ(ierr);

  ERR_HEADER[0] = '\0';
  PetscFunctionReturn(0);
}


PetscErrorCode IOLoadPetscOptions(const char *filename){
  PetscErrorCode ierr;
  const char     **buffer_vals;
  PetscInt       len = 0;
  PetscBool      found;
  char           *copts;

  PetscFunctionBeginUser;
  ierr = IOSeekVarFromLoc(filename, "Options", 0, PETSC_NULL, &found);                  CHKERRQ(ierr);
  if (!found) PetscFunctionReturn(0);
  ierr = PetscSNPrintf(ERR_HEADER, sizeof(ERR_HEADER), "Cannot read Options: ");        CHKERRQ(ierr);
  ierr = IOLoadVarArrayFromLoc(filename, "Options", 0, PETSC_NULL, &len, &buffer_vals); CHKERRQ(ierr);
  ierr = PetscOptionsGetAll(PETSC_NULL, &copts);                                        CHKERRQ(ierr);
  ierr = PetscOptionsClear(PETSC_NULL);                                                 CHKERRQ(ierr);
  for (PetscInt i = 0; i < len; i++){
    ierr = PetscOptionsInsertString(PETSC_NULL, buffer_vals[i]);                        CHKERRQ(ierr);
    ierr = PetscFree(buffer_vals[i]);                                                   CHKERRQ(ierr);
  }
  ierr = PetscOptionsInsertString(PETSC_NULL, copts);                                   CHKERRQ(ierr);
  ierr = PetscFree(buffer_vals);                                                        CHKERRQ(ierr);
  ierr = PetscFree(copts);                                                              CHKERRQ(ierr);
  ERR_HEADER[0] = '\0';
  PetscFunctionReturn(0);
}


PetscErrorCode IOLoadMonitorOptions(const char *filename, const char *name, PetscBool *set, PetscInt *n_iter){
  PetscErrorCode ierr;
  PetscBool      found;
  const char     *buffer_val;

  PetscFunctionBeginUser;
  *set = PETSC_FALSE;
  const char* loc[] = {"MonitorOptions", name};
  ierr = IOSeekVarFromLoc(filename, loc[0], 0, PETSC_NULL, &found); CHKERRQ(ierr);
  if (!found) PetscFunctionReturn(0);
  ierr = IOSeekVarFromLoc(filename, loc[1], 1, loc, &found);        CHKERRQ(ierr);
  if (!found) PetscFunctionReturn(0);
  *set = PETSC_TRUE;

  ierr = PetscSNPrintf(ERR_HEADER, sizeof(ERR_HEADER), "Cannot read MonitorOptions: ");CHKERRQ(ierr);
  ierr = IOLoadVarFromLoc(filename, "n_iter", 2, loc, &buffer_val);                    CHKERRQ(ierr);
  *n_iter = atof(buffer_val);
  ierr = PetscFree(buffer_val);                                                        CHKERRQ(ierr);

  ERR_HEADER[0] = '\0';
  PetscFunctionReturn(0);
}
