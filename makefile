include make.inc

.PHONY: all
all: $(BIN_DIR)/yanss

$(DEP_DIR) $(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# === List of source files
SRC = main.c spatial.c temporal.c physics.c utils.c


$(BIN_DIR)/yanss: $(SRC:%.c=$(OBJ_DIR)/%.o) | $(BIN_DIR)
	@# printf "Linking %-14s " yanss
	$(CC) -o $@ $^ $(LINK_OPTS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(DEP_DIR) $(OBJ_DIR)
	@# printf "Compiling %-12s " $*
	$(CC) -c $(COMP_OPTS) -Iinclude -MMD -MP -MT $@ -MF $(DEP_DIR)/$*.d $< -o $@


-include $(SRC:%.c=$(DEP_DIR)/%.d)


.PHONY: test
test: $(BIN_DIR)/yanss data/box.msh
	$(BIN_DIR)/yanss
data/%.msh: data/%.geo
	gmsh -2 -bin $<


.PHONY: clean clean_dep clean_msh clean_all
clean_all: clean clean_dep clean_msh
clean:
	rm -f $(OBJ_DIR)/*.o $(BIN_DIR)/yanss
clean_dep:
	rm -f $(DEP_DIR)/*.d
clean_msh:
	rm -f data/*.msh
