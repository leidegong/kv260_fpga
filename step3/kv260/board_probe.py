"""Read-only KV260/Linux inventory. No overlay, firmware, memory or settings writes.

Run on the board: python3 board_probe.py --out board_probe.json
This inventory cannot prove DDR bandwidth, correctness or timing closure.
"""
import argparse
import json
from pathlib import Path
import platform


def read_text(path):
    try:
        return Path(path).read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return None


def inventory():
    mem = {}
    for line in (read_text("/proc/meminfo") or "").splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "CmaTotal", "CmaFree", "SwapTotal", "SwapFree"):
            mem[key] = value.strip()
    model = read_text("/proc/device-tree/model")
    compat = read_text("/proc/device-tree/compatible")
    return dict(schema="step3-kv260-probe-v1", system=platform.system(), machine=platform.machine(),
                kernel=platform.release(), os_release=read_text("/etc/os-release"),
                device_tree_model=model, compatible=compat, memory=mem,
                fpga_manager_state=read_text("/sys/class/fpga_manager/fpga0/state"),
                fpga_manager_name=read_text("/sys/class/fpga_manager/fpga0/name"),
                kv260_identified=bool(model and "kv260" in model.lower()),
                bandwidth_measured=False, overlay_loaded_by_this_script=False,
                note="CMA/free RAM does not establish a contiguous 1+ GB DMA allocation; runtime design still required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(inventory(), ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
