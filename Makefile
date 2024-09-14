# Makefile for neural

lib.name = neuralnet

neuralnet.class.sources = src/neuralnet.c
neuralnet~.class.sources = src/neuralnet~.c

cflags = -g -Wno-cast-function-type

datafiles = neuralnet-help.pd neuralnet~-help.pd README.md

PDLIBBUILDER_DIR=./pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder

# Custom target for installing examples
install-examples:
	$(INSTALL_DIR) "$(installpath)/examples"
	find examples -type d -exec $(INSTALL_DIR) "$(installpath)/{}" \;
	find examples -type f -exec $(INSTALL_DATA) "{}" "$(installpath)/{}" \;

# Add custom target to install target
install: install-examples
