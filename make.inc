SHELL = /bin/bash

SRC_DIR = src
DEP_DIR = build/dep
OBJ_DIR = build/obj
BIN_DIR = build/bin

CC = mpicc
COMP_OPTS = -I/home/pierre/.local/include -Wall
LINK_OPTS = -L/home/pierre/.local/lib -lpetsc -lm
