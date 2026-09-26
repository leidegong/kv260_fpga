/* SPDX-License-Identifier: GPL-2.0 OR MIT */
/* Compile-only / -fsyntax-only check that UAPI sizes match the documented layout. */
#include <stdio.h>
#include <stdint.h>
#include "kv260_accel.h"
#include "kv260_addr.h"

#define CHECK_SIZE(t, n) do { \
	if (sizeof(t) != (n)) { \
		fprintf(stderr, #t " size %zu != %d\n", sizeof(t), (n)); \
		return 1; \
	} \
} while (0)

int main(void)
{
	CHECK_SIZE(struct kv260_accel_version, 16);
	CHECK_SIZE(struct kv260_accel_caps, 48);
	CHECK_SIZE(struct kv260_accel_alloc_buffer, 40);
	CHECK_SIZE(struct kv260_accel_free_buffer, 16);
	CHECK_SIZE(struct kv260_accel_set_image_base, 32);
	CHECK_SIZE(struct kv260_accel_submit_desc, 32);
	CHECK_SIZE(struct kv260_accel_probe_info, 32);

	if (kv260_isa_to_phys(0x1000, 1) != 0x1000 + 64)
		return 1;
	if (KV260_ACCEL_REG_CTRL != -1)
		return 1;
	puts("uapi_layout_check OK");
	return 0;
}
