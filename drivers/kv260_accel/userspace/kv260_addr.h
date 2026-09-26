/* SPDX-License-Identifier: GPL-2.0 OR MIT */
/*
 * Userspace helper: ISA addr (64 B units) → device physical/DMA address.
 *
 *   phys = image_base + (isa_addr << 6)
 *
 * image_base must be a DMA device address from the driver (ALLOC_BUFFER /
 * SET_IMAGE_BASE), never a Linux virtual address.
 *
 * 8 KiB software pages are split for AXI by step3/kv260/axi_plan.py
 * (4 KiB boundary / 256 beats). Do not reimplement that split here.
 */
#ifndef KV260_ADDR_H
#define KV260_ADDR_H

#include "kv260_accel.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline uint64_t kv260_isa_to_phys(uint64_t image_base_dma, uint32_t isa_addr)
{
	return kv260_accel_isa_to_phys(image_base_dma, isa_addr);
}

#ifdef __cplusplus
}
#endif

#endif /* KV260_ADDR_H */
