// SPDX-License-Identifier: GPL-2.0
/*
 * kv260_accel — out-of-tree Linux driver skeleton for KV260 Qwen3 PL accel.
 *
 * Intentional limits of this skeleton:
 *  - Probe fails cleanly (-ENODEV / -ENXIO) without a matching DT node /
 *    usable platform device; there is no silent fake success path.
 *  - No hardcoded MMIO physical base. Register offsets in UAPI are TBD (-1).
 *  - Stub MMIO helpers return -ENODEV until resources exist and offsets are
 *    assigned after bitstream + DTS.
 *  - DMA buffers use dma_alloc_coherent; 1 GiB+ contiguous is NOT assumed.
 *  - Does not load bitstreams, claim bandwidth, or report tok/s.
 *
 * Build: see Makefile. Requires ARCH/CROSS_COMPILE (or native aarch64) and
 * a configured kernel tree. Host x86_64 may only syntax-check.
 */

#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/of_device.h>
#include <linux/io.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/dma-mapping.h>
#include <linux/mutex.h>
#include <linux/interrupt.h>
#include <linux/version.h>
#include <linux/bitops.h>

#include "kv260_accel.h"

static struct class *kv260_accel_class;

static struct kv260_accel_buffer *
kv260_find_buffer(struct kv260_accel_dev *adev, u64 handle)
{
	int i;

	for (i = 0; i < KV260_ACCEL_MAX_BUFFERS; i++) {
		if (adev->buffers[i].in_use && adev->buffers[i].handle == handle)
			return &adev->buffers[i];
	}
	return NULL;
}

static long kv260_accel_ioctl(struct file *filp, unsigned int cmd,
			      unsigned long arg)
{
	struct kv260_accel_dev *adev = filp->private_data;
	void __user *uarg = (void __user *)arg;
	int ret = 0;

	if (!adev)
		return -ENODEV;

	switch (cmd) {
	case KV260_ACCEL_IOCTL_GET_VERSION: {
		struct kv260_accel_version ver = {
			.major = KV260_ACCEL_DRIVER_VERSION_MAJOR,
			.minor = KV260_ACCEL_DRIVER_VERSION_MINOR,
			.patch = KV260_ACCEL_DRIVER_VERSION_PATCH,
			.abi = KV260_ACCEL_ABI_VERSION,
		};

		if (copy_to_user(uarg, &ver, sizeof(ver)))
			return -EFAULT;
		return 0;
	}
	case KV260_ACCEL_IOCTL_QUERY_CAPS: {
		struct kv260_accel_caps caps;

		memset(&caps, 0, sizeof(caps));
		mutex_lock(&adev->lock);
		caps.flags = adev->caps_flags;
		if (adev->image_base_set)
			caps.flags |= KV260_ACCEL_CAP_IMAGE_BASE_SET;
		caps.max_buffers = KV260_ACCEL_MAX_BUFFERS;
		caps.isa_addr_shift = KV260_ACCEL_ISA_ADDR_SHIFT;
		caps.max_buffer_bytes = KV260_ACCEL_MAX_BUFFER_BYTES;
		caps.page_bytes = KV260_ACCEL_SOFT_PAGE_BYTES;
		caps.features = 0;
		caps.image_base_dma = adev->image_base_set ? adev->image_base_dma : 0;
		mutex_unlock(&adev->lock);
		if (copy_to_user(uarg, &caps, sizeof(caps)))
			return -EFAULT;
		return 0;
	}
	case KV260_ACCEL_IOCTL_ALLOC_BUFFER: {
		struct kv260_accel_alloc_buffer req;
		struct kv260_accel_buffer *buf = NULL;
		int i;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;
		if (!req.size || req.size > KV260_ACCEL_MAX_BUFFER_BYTES)
			return -EINVAL;
		if (req.flags & ~(KV260_ACCEL_BUF_DMA_COHERENT |
				  KV260_ACCEL_BUF_DIRECTION_BIDI))
			return -EINVAL;

		mutex_lock(&adev->lock);
		for (i = 0; i < KV260_ACCEL_MAX_BUFFERS; i++) {
			if (!adev->buffers[i].in_use) {
				buf = &adev->buffers[i];
				break;
			}
		}
		if (!buf) {
			mutex_unlock(&adev->lock);
			return -ENOMEM;
		}
		/*
		 * dma_alloc_coherent needs a DMA-capable device. Without a real
		 * DT/platform DMA setup this returns failure — do not fake success.
		 */
		buf->cpu_addr = dma_alloc_coherent(adev->dev, req.size,
						   &buf->dma_addr, GFP_KERNEL);
		if (!buf->cpu_addr) {
			mutex_unlock(&adev->lock);
			return -ENOMEM;
		}
		buf->size = req.size;
		buf->handle = ++adev->next_handle;
		buf->in_use = true;
		req.handle = buf->handle;
		req.dma_addr = (u64)buf->dma_addr;
		req.cpu_mmap_offset = 0;
		adev->caps_flags |= KV260_ACCEL_CAP_HAS_DMA;
		mutex_unlock(&adev->lock);

		if (copy_to_user(uarg, &req, sizeof(req))) {
			mutex_lock(&adev->lock);
			dma_free_coherent(adev->dev, buf->size, buf->cpu_addr,
					  buf->dma_addr);
			buf->in_use = false;
			buf->cpu_addr = NULL;
			mutex_unlock(&adev->lock);
			return -EFAULT;
		}
		return 0;
	}
	case KV260_ACCEL_IOCTL_FREE_BUFFER: {
		struct kv260_accel_free_buffer req;
		struct kv260_accel_buffer *buf;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;
		if (req.reserved)
			return -EINVAL;
		mutex_lock(&adev->lock);
		buf = kv260_find_buffer(adev, req.handle);
		if (!buf) {
			mutex_unlock(&adev->lock);
			return -EINVAL;
		}
		dma_free_coherent(adev->dev, buf->size, buf->cpu_addr,
				  buf->dma_addr);
		if (adev->image_base_set &&
		    adev->image_base_dma == (u64)buf->dma_addr) {
			adev->image_base_set = false;
			adev->image_base_dma = 0;
		}
		memset(buf, 0, sizeof(*buf));
		mutex_unlock(&adev->lock);
		return 0;
	}
	case KV260_ACCEL_IOCTL_SET_IMAGE_BASE: {
		struct kv260_accel_set_image_base req;
		struct kv260_accel_buffer *buf;
		u64 dma;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;
		if (req.flags || req.reserved)
			return -EINVAL;

		mutex_lock(&adev->lock);
		if (req.handle) {
			buf = kv260_find_buffer(adev, req.handle);
			if (!buf) {
				mutex_unlock(&adev->lock);
				return -EINVAL;
			}
			dma = (u64)buf->dma_addr;
		} else if (req.dma_addr) {
			/* Explicit DMA address: caller must own a mapping. */
			dma = req.dma_addr;
		} else {
			mutex_unlock(&adev->lock);
			return -EINVAL;
		}
		adev->image_base_dma = dma;
		adev->image_base_set = true;
		/*
		 * Programming IMAGE_BASE into PL regs is TBD: stub returns
		 * success for software state only; MMIO write would be -ENODEV.
		 */
		(void)kv260_accel_mmio_write32(adev, KV260_ACCEL_REG_IMAGE_BASE_LO,
					       lower_32_bits(dma));
		(void)kv260_accel_mmio_write32(adev, KV260_ACCEL_REG_IMAGE_BASE_HI,
					       upper_32_bits(dma));
		mutex_unlock(&adev->lock);
		return 0;
	}
	case KV260_ACCEL_IOCTL_SUBMIT_DESC: {
		struct kv260_accel_submit_desc req;

		if (copy_from_user(&req, uarg, sizeof(req)))
			return -EFAULT;
		/* No PL control path yet. */
		req.status = -ENODEV;
		if (copy_to_user(uarg, &req, sizeof(req)))
			return -EFAULT;
		return -ENODEV;
	}
	case KV260_ACCEL_IOCTL_RUN_PROBE: {
		struct kv260_accel_probe_info info;

		memset(&info, 0, sizeof(info));
		mutex_lock(&adev->lock);
		info.flags = adev->caps_flags;
		if (adev->image_base_set)
			info.flags |= KV260_ACCEL_CAP_IMAGE_BASE_SET;
		info.kernel_has_cma = 0; /* filled only when /proc or DT CMA known; TBD */
		info.note_flags = KV260_ACCEL_NOTE_NO_BANDWIDTH |
				  KV260_ACCEL_NOTE_NO_BITSTREAM |
				  KV260_ACCEL_NOTE_SG_TBD;
		mutex_unlock(&adev->lock);
		if (copy_to_user(uarg, &info, sizeof(info)))
			return -EFAULT;
		return 0;
	}
	default:
		ret = -ENOTTY;
		break;
	}
	return ret;
}

static int kv260_accel_open(struct inode *inode, struct file *filp)
{
	struct kv260_accel_dev *adev =
		container_of(inode->i_cdev, struct kv260_accel_dev, cdev);

	filp->private_data = adev;
	return 0;
}

static int kv260_accel_release(struct inode *inode, struct file *filp)
{
	filp->private_data = NULL;
	return 0;
}

static const struct file_operations kv260_accel_fops = {
	.owner = THIS_MODULE,
	.open = kv260_accel_open,
	.release = kv260_accel_release,
	.unlocked_ioctl = kv260_accel_ioctl,
	.compat_ioctl = kv260_accel_ioctl,
	.llseek = no_llseek,
};

static int kv260_accel_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct kv260_accel_dev *adev;
	struct resource *res;
	int ret;

	adev = devm_kzalloc(dev, sizeof(*adev), GFP_KERNEL);
	if (!adev)
		return -ENOMEM;

	adev->dev = dev;
	adev->pdev = pdev;
	adev->irq = -1;
	mutex_init(&adev->lock);
	adev->next_handle = 1;
	adev->caps_flags = KV260_ACCEL_CAP_HAS_DT;

	/*
	 * Resources are optional until bitstream + DTS assign them.
	 * Missing reg/irq is OK for early bring-up, but we still refuse to
	 * pretend MMIO works. If OF match fired we have a node; continue.
	 */
	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (res) {
		/*
		 * Do not ioremap until offsets are real. Mapping a present but
		 * TBD region without a known register map is still unsafe —
		 * record size only and leave has_reg false until bring-up.
		 */
		adev->regs_size = resource_size(res);
		dev_info(dev,
			 "reg resource present size=0x%llx but MMIO map deferred (offsets TBD)\n",
			 (unsigned long long)adev->regs_size);
		adev->has_reg = false;
	} else {
		dev_warn(dev, "no reg resource yet (bitstream/DTS TBD); MMIO stubs return -ENODEV\n");
		adev->has_reg = false;
	}

	ret = platform_get_irq_optional(pdev, 0);
	if (ret > 0) {
		adev->irq = ret;
		adev->has_irq = true;
		adev->caps_flags |= KV260_ACCEL_CAP_HAS_IRQ;
		/* Handler not installed until PL IRQ semantics exist. */
		dev_info(dev, "irq %d present but handler deferred\n", adev->irq);
	} else {
		adev->has_irq = false;
	}

	ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(64));
	if (ret) {
		ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(32));
		if (ret) {
			dev_err(dev, "DMA mask setup failed: %d\n", ret);
			return ret;
		}
	}

	ret = alloc_chrdev_region(&adev->devt, 0, 1, KV260_ACCEL_NAME);
	if (ret) {
		dev_err(dev, "alloc_chrdev_region failed: %d\n", ret);
		return ret;
	}

	cdev_init(&adev->cdev, &kv260_accel_fops);
	adev->cdev.owner = THIS_MODULE;
	ret = cdev_add(&adev->cdev, adev->devt, 1);
	if (ret) {
		dev_err(dev, "cdev_add failed: %d\n", ret);
		goto err_unreg;
	}

	adev->class = kv260_accel_class;
	adev->char_dev = device_create(kv260_accel_class, dev, adev->devt, adev,
				       KV260_ACCEL_NAME);
	if (IS_ERR(adev->char_dev)) {
		ret = PTR_ERR(adev->char_dev);
		dev_err(dev, "device_create failed: %d\n", ret);
		goto err_cdev;
	}

	platform_set_drvdata(pdev, adev);
	dev_info(dev,
		 "kv260_accel skeleton probed (reg_mapped=%d irq=%d); not board-validated\n",
		 adev->has_reg, adev->has_irq);
	return 0;

err_cdev:
	cdev_del(&adev->cdev);
err_unreg:
	unregister_chrdev_region(adev->devt, 1);
	return ret;
}

static void kv260_accel_remove(struct platform_device *pdev)
{
	struct kv260_accel_dev *adev = platform_get_drvdata(pdev);
	int i;

	if (!adev)
		return;

	mutex_lock(&adev->lock);
	for (i = 0; i < KV260_ACCEL_MAX_BUFFERS; i++) {
		if (!adev->buffers[i].in_use)
			continue;
		dma_free_coherent(adev->dev, adev->buffers[i].size,
				  adev->buffers[i].cpu_addr,
				  adev->buffers[i].dma_addr);
		adev->buffers[i].in_use = false;
	}
	mutex_unlock(&adev->lock);

	device_destroy(kv260_accel_class, adev->devt);
	cdev_del(&adev->cdev);
	unregister_chrdev_region(adev->devt, 1);
	/* regs never mapped in skeleton */
	adev->regs = NULL;
	dev_info(&pdev->dev, "kv260_accel removed\n");
}

static const struct of_device_id kv260_accel_of_match[] = {
	{ .compatible = KV260_ACCEL_COMPATIBLE },
	{ /* sentinel */ }
};
MODULE_DEVICE_TABLE(of, kv260_accel_of_match);

static struct platform_driver kv260_accel_driver = {
	.probe = kv260_accel_probe,
	.remove = kv260_accel_remove,
	.driver = {
		.name = KV260_ACCEL_NAME,
		.of_match_table = kv260_accel_of_match,
	},
};

static int __init kv260_accel_init(void)
{
	int ret;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
	kv260_accel_class = class_create(KV260_ACCEL_NAME);
#else
	kv260_accel_class = class_create(THIS_MODULE, KV260_ACCEL_NAME);
#endif
	if (IS_ERR(kv260_accel_class)) {
		pr_err("kv260_accel: class_create failed\n");
		return PTR_ERR(kv260_accel_class);
	}

	ret = platform_driver_register(&kv260_accel_driver);
	if (ret) {
		class_destroy(kv260_accel_class);
		kv260_accel_class = NULL;
		pr_err("kv260_accel: platform_driver_register failed: %d\n", ret);
		return ret;
	}
	/*
	 * Without a matching DT node, probe never runs and /dev/kv260_accel
	 * is not created — userspace open() fails. That is intentional.
	 */
	pr_info("kv260_accel: module loaded (skeleton; waiting for DT node %s)\n",
		KV260_ACCEL_COMPATIBLE);
	return 0;
}

static void __exit kv260_accel_exit(void)
{
	platform_driver_unregister(&kv260_accel_driver);
	if (kv260_accel_class) {
		class_destroy(kv260_accel_class);
		kv260_accel_class = NULL;
	}
	pr_info("kv260_accel: module unloaded\n");
}

module_init(kv260_accel_init);
module_exit(kv260_accel_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("KV260 FPGA Qwen3 project");
MODULE_DESCRIPTION("KV260 PL accelerator driver skeleton (not board-validated)");
MODULE_VERSION("0.1.0");
