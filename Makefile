# Makefile for neuralnet

lib.name = neuralnet

neuralnet.class.sources = src/neuralnet.c
neuralnet~.class.sources = src/neuralnet~.c

cflags = -g -Wno-cast-function-type

datafiles = neuralnet-help.pd neuralnet~-help.pd README.md

# Try to find pd-lib-builder
PDLIBBUILDER_DIR ?= $(firstword $(wildcard pd-lib-builder) $(wildcard ../pd-lib-builder))

ifneq ($(PDLIBBUILDER_DIR),)
  include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder
else
  $(error pd-lib-builder not found. Please specify path)
endif

# Custom target for installing examples
install-examples:
	$(INSTALL_DIR) "$(installpath)/examples"
	find examples -type d -exec $(INSTALL_DIR) "$(installpath)/{}" \;
	find examples -type f -exec $(INSTALL_DATA) "{}" "$(installpath)/{}" \;

# Add custom target to install target
install: install-examples
