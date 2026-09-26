/* SPDX-License-Identifier: GPL-2.0 OR MIT */
/*
 * Example CLI for /dev/kv260_accel.
 * Off-board: open() is expected to fail (ENOENT/ENODEV). --selftest exercises
 * local helpers and ioctl number packing without a device node.
 */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include "kv260_accel.h"
#include "kv260_addr.h"

static const char *devpath = "/dev/kv260_accel";

static int do_selftest(void)
{
	uint64_t phys;
	int fails = 0;

	/* Vectors: image_base + (addr << 6) */
	struct {
		uint64_t base;
		uint32_t addr;
		uint64_t expect;
	} vecs[] = {
		{ 0x0, 0, 0x0 },
		{ 0x1000, 1, 0x1000 + 64 },
		{ 0x80000000ULL, 0x10, 0x80000000ULL + (0x10ULL << 6) },
		{ 0x100000000ULL, 0xffffffffu / 64, /* large */ 0 },
	};
	/* Fix last expect explicitly */
	vecs[3].expect = 0x100000000ULL + ((uint64_t)(0xffffffffu / 64) << 6);

	for (size_t i = 0; i < sizeof(vecs) / sizeof(vecs[0]); i++) {
		phys = kv260_isa_to_phys(vecs[i].base, vecs[i].addr);
		if (phys != vecs[i].expect) {
			fprintf(stderr, "selftest addr fail [%zu]: got 0x%" PRIx64
				" expect 0x%" PRIx64 "\n",
				i, phys, vecs[i].expect);
			fails++;
		}
	}

	if (sizeof(struct kv260_accel_version) != 16) {
		fprintf(stderr, "version size %zu != 16\n",
			sizeof(struct kv260_accel_version));
		fails++;
	}
	if (sizeof(struct kv260_accel_caps) != 48) {
		fprintf(stderr, "caps size %zu != 48\n",
			sizeof(struct kv260_accel_caps));
		fails++;
	}
	if (sizeof(struct kv260_accel_alloc_buffer) != 40) {
		fprintf(stderr, "alloc size %zu != 40\n",
			sizeof(struct kv260_accel_alloc_buffer));
		fails++;
	}
	if (KV260_ACCEL_ISA_ADDR_SHIFT != 6 || KV260_ACCEL_SOFT_PAGE_BYTES != 8192) {
		fprintf(stderr, "constant mismatch\n");
		fails++;
	}
	if (KV260_ACCEL_REG_CTRL != -1) {
		fprintf(stderr, "REG_CTRL must remain TBD (-1)\n");
		fails++;
	}

	/* ioctl macros must be non-zero */
	if (!KV260_ACCEL_IOCTL_GET_VERSION || !KV260_ACCEL_IOCTL_QUERY_CAPS ||
	    !KV260_ACCEL_IOCTL_ALLOC_BUFFER || !KV260_ACCEL_IOCTL_SET_IMAGE_BASE) {
		fprintf(stderr, "ioctl macros look wrong\n");
		fails++;
	}

	if (fails) {
		fprintf(stderr, "kv260_cli --selftest: %d failure(s)\n", fails);
		return 1;
	}
	printf("kv260_cli --selftest: OK (struct layout + isa_to_phys)\n");
	printf("  note: AXI 8KiB page split lives in step3/kv260/axi_plan.py\n");
	return 0;
}

static int run_device(void)
{
	int fd;
	struct kv260_accel_version ver;
	struct kv260_accel_caps caps;
	struct kv260_accel_alloc_buffer alloc;
	struct kv260_accel_free_buffer freeb;
	struct kv260_accel_probe_info probe;

	fd = open(devpath, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		fprintf(stderr,
			"open(%s) failed: %s\n"
			"Expected off-board (no DT node / module). "
			"On KV260 after DTS + insmod, re-run this CLI.\n",
			devpath, strerror(errno));
		return 2;
	}

	memset(&ver, 0, sizeof(ver));
	if (ioctl(fd, KV260_ACCEL_IOCTL_GET_VERSION, &ver) < 0) {
		fprintf(stderr, "GET_VERSION: %s\n", strerror(errno));
		close(fd);
		return 1;
	}
	printf("version %u.%u.%u abi=%u\n", ver.major, ver.minor, ver.patch, ver.abi);

	memset(&caps, 0, sizeof(caps));
	if (ioctl(fd, KV260_ACCEL_IOCTL_QUERY_CAPS, &caps) < 0) {
		fprintf(stderr, "QUERY_CAPS: %s\n", strerror(errno));
		close(fd);
		return 1;
	}
	printf("caps flags=0x%x max_buffers=%u max_buffer_bytes=%" PRIu64
	       " page=%" PRIu64 " isa_shift=%u image_base=0x%" PRIx64 "\n",
	       caps.flags, caps.max_buffers, (uint64_t)caps.max_buffer_bytes,
	       (uint64_t)caps.page_bytes, caps.isa_addr_shift,
	       (uint64_t)caps.image_base_dma);

	memset(&probe, 0, sizeof(probe));
	if (ioctl(fd, KV260_ACCEL_IOCTL_RUN_PROBE, &probe) < 0) {
		fprintf(stderr, "RUN_PROBE: %s\n", strerror(errno));
	} else {
		printf("probe flags=0x%x note_flags=0x%x (no bandwidth claim)\n",
		       probe.flags, probe.note_flags);
	}

	memset(&alloc, 0, sizeof(alloc));
	alloc.size = 4096;
	alloc.flags = KV260_ACCEL_BUF_DMA_COHERENT;
	if (ioctl(fd, KV260_ACCEL_IOCTL_ALLOC_BUFFER, &alloc) < 0) {
		fprintf(stderr, "ALLOC_BUFFER: %s (may fail without DMA/CMA)\n",
			strerror(errno));
	} else {
		printf("alloc handle=%" PRIu64 " dma=0x%" PRIx64 "\n",
		       (uint64_t)alloc.handle, (uint64_t)alloc.dma_addr);
		printf("  example phys for isa_addr=2: 0x%" PRIx64 "\n",
		       kv260_isa_to_phys(alloc.dma_addr, 2));
		memset(&freeb, 0, sizeof(freeb));
		freeb.handle = alloc.handle;
		if (ioctl(fd, KV260_ACCEL_IOCTL_FREE_BUFFER, &freeb) < 0)
			fprintf(stderr, "FREE_BUFFER: %s\n", strerror(errno));
	}

	close(fd);
	return 0;
}

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--selftest"))
			return do_selftest();
		if (!strcmp(argv[i], "--device") && i + 1 < argc) {
			devpath = argv[++i];
			continue;
		}
		if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
			printf("Usage: %s [--selftest] [--device PATH]\n", argv[0]);
			return 0;
		}
	}
	return run_device();
}
