# kv260_accel out-of-tree driver + userspace stubs
#
# Kernel module (on KV260 / cross build):
#   make module KERNELDIR=/path/to/kria-linux ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-
#
# Host (this repo box, typically x86_64, no board DT):
#   make test          # UAPI layout + address helper (no kernel headers)
#   make userspace     # build CLI
#   make syntax-check  # gcc -fsyntax-only on userspace sources
#
# Without a configured kernel tree, "make module" exits 2. That is OK.

DRIVER_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
KERNELDIR ?= /lib/modules/$(shell uname -r)/build
ARCH ?= $(shell uname -m | sed 's/x86_64/x86/;s/aarch64/arm64/')
CROSS_COMPILE ?=

CC ?= gcc
CFLAGS_USER ?= -Wall -Wextra -O2 -I$(DRIVER_DIR)/uapi -I$(DRIVER_DIR)/userspace
PYTHON ?= python3

.PHONY: all test userspace syntax-check module clean help

all: userspace test

help:
	@echo "Targets: userspace test syntax-check module clean"
	@echo "Module needs KERNELDIR + matching ARCH/CROSS_COMPILE for the board."
	@echo "Compatible string: liaojiang,kv260-accel"

userspace: userspace/kv260_cli tests/uapi_layout_check

userspace/kv260_cli: userspace/kv260_cli.c uapi/kv260_accel.h userspace/kv260_addr.h
	$(CC) $(CFLAGS_USER) -o $@ userspace/kv260_cli.c

tests/uapi_layout_check: tests/uapi_layout_check.c uapi/kv260_accel.h userspace/kv260_addr.h
	$(CC) $(CFLAGS_USER) -o $@ tests/uapi_layout_check.c

syntax-check:
	$(CC) $(CFLAGS_USER) -fsyntax-only userspace/kv260_cli.c
	$(CC) $(CFLAGS_USER) -fsyntax-only tests/uapi_layout_check.c

test: userspace syntax-check
	$(PYTHON) tests/test_uapi.py
	./tests/uapi_layout_check
	./userspace/kv260_cli --selftest
	@echo "kv260_accel host tests OK (no board required)"

module:
	@if [ ! -f "$(KERNELDIR)/Makefile" ]; then \
		echo "KERNELDIR=$(KERNELDIR) is not a configured kernel tree."; \
		echo "Install linux-headers or point KERNELDIR at the Kria kernel."; \
		echo "Skipping module build (optional on host)."; \
		exit 2; \
	fi
	$(MAKE) -C $(KERNELDIR) M=$(DRIVER_DIR) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) modules

clean:
	-$(MAKE) -C $(KERNELDIR) M=$(DRIVER_DIR) clean 2>/dev/null
	rm -f userspace/kv260_cli tests/uapi_layout_check
	rm -f *.o *.ko *.mod *.mod.c .*.cmd modules.order Module.symvers
	rm -rf .tmp_versions
