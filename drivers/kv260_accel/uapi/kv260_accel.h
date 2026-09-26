/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Userspace API for the KV260 FPGA Qwen3 accelerator skeleton.
 *
 * This is an ABI *draft*. Register / MMIO offsets are not assigned until a
 * bitstream and device-tree binding exist. Do not treat any value in this
 * header as a verified physical base address or measured performance number.
 *
 * Address contract (Step3 KV260 实施规格 §4):
 *   image offsets are bytes;
 *   ISA addr fields are 64-byte units;
 *   physical = IMAGE_BASE + (instruction.addr << 6);
 *   IMAGE_BASE is a DMA device address from ALLOC_BUFFER / SET_IMAGE_BASE,
 *   never a Linux virtual address and never a hardcoded MMIO base.
 */
#ifndef _UAPI_KV260_ACCEL_H
#define _UAPI_KV260_ACCEL_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
typedef uint8_t  __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef uint64_t __u64;
typedef int32_t  __s32;
typedef int64_t  __s64;
#endif

#define KV260_ACCEL_IOCTL_MAGIC		0x4B /* 'K' */
#define KV260_ACCEL_ABI_VERSION		1
#define KV260_ACCEL_DRIVER_VERSION_MAJOR	0
#define KV260_ACCEL_DRIVER_VERSION_MINOR	1
#define KV260_ACCEL_DRIVER_VERSION_PATCH	0

/* ISA address unit: each addr field counts 64-byte granules. */
#define KV260_ACCEL_ISA_ADDR_SHIFT	6
#define KV260_ACCEL_ISA_ADDR_UNIT	(1u << KV260_ACCEL_ISA_ADDR_SHIFT)

/* Software page size used by step3 packing (see step3/kv260/axi_plan.py). */
#define KV260_ACCEL_SOFT_PAGE_BYTES	8192u
#define KV260_ACCEL_AXI_BOUNDARY_BYTES	4096u
#define KV260_ACCEL_AXI_MAX_BEATS	256u

/*
 * Named MMIO / register offset placeholders.
 * Values are TBD (-1) until bitstream + DTS assign them. Stub MMIO helpers
 * must return -ENODEV while resources are absent; never map a fake base.
 */
#define KV260_ACCEL_REG_CTRL		(-1)
#define KV260_ACCEL_REG_STATUS		(-1)
#define KV260_ACCEL_REG_IMAGE_BASE_LO	(-1)
#define KV260_ACCEL_REG_IMAGE_BASE_HI	(-1)
#define KV260_ACCEL_REG_IRQ_MASK	(-1)
#define KV260_ACCEL_REG_IRQ_STATUS	(-1)

/* Capability / readiness flags (also used by RUN_PROBE). */
#define KV260_ACCEL_CAP_HAS_DT		(1u << 0) /* matching DT node seen */
#define KV260_ACCEL_CAP_HAS_REG		(1u << 1) /* reg resource mapped */
#define KV260_ACCEL_CAP_HAS_IRQ		(1u << 2) /* interrupt wired */
#define KV260_ACCEL_CAP_HAS_DMA		(1u << 3) /* DMA device usable */
#define KV260_ACCEL_CAP_IMAGE_BASE_SET	(1u << 4) /* SET_IMAGE_BASE applied */
#define KV260_ACCEL_CAP_BITSTREAM_HINT	(1u << 5) /* optional firmware name present; NOT proof of load */
#define KV260_ACCEL_CAP_BOARD_PROBE_OK	(1u << 6) /* host-side probe fields readable */

/* Buffer allocation flags. */
#define KV260_ACCEL_BUF_DMA_COHERENT	(1u << 0) /* dma_alloc_coherent path */
#define KV260_ACCEL_BUF_DIRECTION_BIDI	(1u << 1)

/**
 * struct kv260_accel_version - driver / ABI version
 * @major/@minor/@patch: driver version
 * @abi: UAPI ABI version (KV260_ACCEL_ABI_VERSION)
 */
struct kv260_accel_version {
	__u32 major;
	__u32 minor;
	__u32 patch;
	__u32 abi;
};

/**
 * struct kv260_accel_caps - QUERY_CAPS result
 * @flags: KV260_ACCEL_CAP_* readiness bits (skeleton: mostly clear off-board)
 * @reserved0: pad / future
 * @max_buffers: soft limit for concurrent ALLOC_BUFFER handles
 * @max_buffer_bytes: per-buffer soft limit (NOT a promise of 1 GiB+ contiguous CMA)
 * @page_bytes: soft page size (8192); AXI split is documented in axi_plan.py
 * @isa_addr_shift: must be 6
 * @features: reserved feature bits (0 in skeleton)
 * @image_base_dma: current IMAGE_BASE device address, or 0 if unset
 *
 * Limitations: do not assume one allocation yields a contiguous 1+ GiB image.
 * Final runtime may use scatter-gather / segmented DMA or reserved memory (TBD).
 */
struct kv260_accel_caps {
	__u32 flags;
	__u32 reserved0;
	__u32 max_buffers;
	__u32 isa_addr_shift;
	__u64 max_buffer_bytes;
	__u64 page_bytes;
	__u64 features;
	__u64 image_base_dma;
};

/**
 * struct kv260_accel_alloc_buffer - ALLOC_BUFFER
 * @size: requested bytes (in)
 * @flags: KV260_ACCEL_BUF_* (in)
 * @handle: opaque buffer id (out)
 * @dma_addr: device (AXI) address suitable as IMAGE_BASE or descriptor target (out)
 * @cpu_mmap_offset: reserved for future mmap offset (out, 0 in skeleton)
 *
 * Approach: DMA-coherent buffers via dma_alloc_coherent (not dma-buf export yet).
 * Cache coherency: coherent alloc avoids explicit sync on this path; non-coherent
 * / streaming DMA is TBD and must document sync points when added.
 */
struct kv260_accel_alloc_buffer {
	__u64 size;
	__u64 flags;
	__u64 handle;
	__u64 dma_addr;
	__u64 cpu_mmap_offset;
};

/**
 * struct kv260_accel_free_buffer - FREE_BUFFER
 * @handle: id from ALLOC_BUFFER
 * @reserved: must be 0
 */
struct kv260_accel_free_buffer {
	__u64 handle;
	__u64 reserved;
};

/**
 * struct kv260_accel_set_image_base - SET_IMAGE_BASE
 * @handle: buffer whose dma_addr becomes IMAGE_BASE (preferred)
 * @dma_addr: explicit device address if handle==0 (must be DMA-mapped)
 * @flags: reserved, must be 0
 * @reserved: must be 0
 *
 * Hardware physical address for ISA ops:
 *   phys = image_base_dma + ((uint64_t)isa_addr << KV260_ACCEL_ISA_ADDR_SHIFT)
 */
struct kv260_accel_set_image_base {
	__u64 handle;
	__u64 dma_addr;
	__u64 flags;
	__u64 reserved;
};

/**
 * struct kv260_accel_submit_desc - SUBMIT_DESC (placeholder)
 * @handle: descriptor / command buffer handle (TBD layout)
 * @nbytes: valid bytes in buffer
 * @flags: reserved
 * @status: driver-filled result; skeleton returns -ENODEV if no MMIO
 *
 * Real descriptor format tracks DCU/ISA binary streams from step3/export_kv260.py.
 * Not implemented beyond ABI shape until PL control path exists.
 */
struct kv260_accel_submit_desc {
	__u64 handle;
	__u64 nbytes;
	__u64 flags;
	__s32 status;
	__u32 reserved;
};

/**
 * struct kv260_accel_probe_info - RUN_PROBE (readiness mirror of board_probe.py)
 * @flags: KV260_ACCEL_CAP_* subset filled by driver
 * @kernel_has_cma: 1 if CMA appears present (informational only)
 * @reserved0: pad
 * @note_flags: informational bits; never claims measured bandwidth/tok/s
 *
 * Mirrors readiness fields only. bandwidth_measured stays false by design.
 */
struct kv260_accel_probe_info {
	__u32 flags;
	__u32 kernel_has_cma;
	__u32 note_flags;
	__u32 reserved0;
	__u64 reserved1;
	__u64 reserved2;
};

#define KV260_ACCEL_NOTE_NO_BANDWIDTH	(1u << 0)
#define KV260_ACCEL_NOTE_NO_BITSTREAM	(1u << 1)
#define KV260_ACCEL_NOTE_SG_TBD		(1u << 2)

#define KV260_ACCEL_IOCTL_GET_VERSION \
	_IOR(KV260_ACCEL_IOCTL_MAGIC, 0x01, struct kv260_accel_version)
#define KV260_ACCEL_IOCTL_QUERY_CAPS \
	_IOR(KV260_ACCEL_IOCTL_MAGIC, 0x02, struct kv260_accel_caps)
#define KV260_ACCEL_IOCTL_ALLOC_BUFFER \
	_IOWR(KV260_ACCEL_IOCTL_MAGIC, 0x03, struct kv260_accel_alloc_buffer)
#define KV260_ACCEL_IOCTL_FREE_BUFFER \
	_IOW(KV260_ACCEL_IOCTL_MAGIC, 0x04, struct kv260_accel_free_buffer)
#define KV260_ACCEL_IOCTL_SET_IMAGE_BASE \
	_IOW(KV260_ACCEL_IOCTL_MAGIC, 0x05, struct kv260_accel_set_image_base)
#define KV260_ACCEL_IOCTL_SUBMIT_DESC \
	_IOWR(KV260_ACCEL_IOCTL_MAGIC, 0x06, struct kv260_accel_submit_desc)
#define KV260_ACCEL_IOCTL_RUN_PROBE \
	_IOR(KV260_ACCEL_IOCTL_MAGIC, 0x07, struct kv260_accel_probe_info)

static inline __u64 kv260_accel_isa_to_phys(__u64 image_base_dma, __u32 isa_addr)
{
	return image_base_dma + ((__u64)isa_addr << KV260_ACCEL_ISA_ADDR_SHIFT);
}

#endif /* _UAPI_KV260_ACCEL_H */
