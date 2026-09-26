/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Internal header for kv260_accel out-of-tree skeleton.
 * MMIO register symbols are TBD placeholders from uapi/kv260_accel.h.
 */
#ifndef _KV260_ACCEL_H
#define _KV260_ACCEL_H

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/mutex.h>
#include <linux/platform_device.h>
#include <linux/types.h>

#include "uapi/kv260_accel.h"

#define KV260_ACCEL_NAME		"kv260_accel"
#define KV260_ACCEL_COMPATIBLE		"liaojiang,kv260-accel"
#define KV260_ACCEL_MAX_BUFFERS		16
/* Soft per-buffer cap for the skeleton; not a CMA/phys contiguity guarantee. */
#define KV260_ACCEL_MAX_BUFFER_BYTES	(256ull << 20)

struct kv260_accel_buffer {
	void *cpu_addr;
	dma_addr_t dma_addr;
	size_t size;
	u64 handle;
	bool in_use;
};

struct kv260_accel_dev {
	struct device *dev;
	struct platform_device *pdev;
	struct cdev cdev;
	struct class *class;
	dev_t devt;
	struct device *char_dev;
	void __iomem *regs;
	resource_size_t regs_size;
	int irq;
	bool has_reg;
	bool has_irq;
	u32 caps_flags;
	u64 image_base_dma;
	bool image_base_set;
	struct mutex lock;
	struct kv260_accel_buffer buffers[KV260_ACCEL_MAX_BUFFERS];
	u64 next_handle;
};

/* Stub MMIO: always -ENODEV until a real reg resource is mapped. */
static inline int kv260_accel_mmio_read32(struct kv260_accel_dev *adev,
					 int reg_offset, u32 *out)
{
	if (!adev || !adev->has_reg || !adev->regs || reg_offset < 0)
		return -ENODEV;
	/* Offset table is TBD; refuse until assigned after bitstream + DTS. */
	(void)out;
	return -ENODEV;
}

static inline int kv260_accel_mmio_write32(struct kv260_accel_dev *adev,
					  int reg_offset, u32 value)
{
	if (!adev || !adev->has_reg || !adev->regs || reg_offset < 0)
		return -ENODEV;
	(void)value;
	return -ENODEV;
}

#endif /* _KV260_ACCEL_H */
