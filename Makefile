# Makefile for neural

lib.name = neuralnet

neuralnet.class.sources = src/neuralnet.c
neuralnet~.class.sources = src/neuralnet~.c

cflags = -g -Wno-cast-function-type

datafiles = neuralnet-help.pd neuralnet~-help.pd README.md

PDLIBBUILDER_DIR=../pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder
