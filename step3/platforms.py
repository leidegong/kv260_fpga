"""Board facts and explicit, unmeasured KV260 implementation assumptions.

The DDR controller belongs to the PS. PL bandwidth is limited by BOTH the
external DDR bus and the enabled HP interfaces. No P3 DSP packing is assumed.
"""
from dataclasses import dataclass
import math

from perf_model import Platform


SOURCES = {
    "memory": "https://docs.amd.com/r/en-US/ds987-k26-som/Functional-Overview-and-Block-Diagram",
    "resources": "https://docs.amd.com/r/en-US/ds987-k26-som/Programmable-Logic",
    "dsp": "https://docs.amd.com/v/u/en-US/ug579-ultrascale-dsp",
    "ports": "https://arxiv.org/html/2507.03308v2#S2.SS3",
    "board_flow": "https://docs.amd.com/r/en-US/ug1089-kv260-starter-kit/Vivado-Board-Flow",
}


@dataclass(frozen=True)
class BoardResources:
    name: str = "KV260 / K26"
    ddr_bytes: int = 4 * 2**30
    ddr_rate_mts: int = 2400
    ddr_bus_bits: int = 64
    lut: int = 117120
    ff: int = 234240
    bram36: int = 144
    uram288: int = 64
    dsp48e2: int = 1248

    @property
    def ddr_peak_gbs(self):
        return self.ddr_rate_mts * self.ddr_bus_bits / 8000

    @property
    def ram_payload_bytes(self):
        """Common 32/64-bit payload widths; excludes parity and distributed RAM."""
        return self.bram36 * 4096 + self.uram288 * 32768


KV260_BOARD = BoardResources()


@dataclass(frozen=True)
class KV260Config:
    """Proposed design, NOT achieved clocks/utilization or a bitstream description.

    Efficiencies are ceilings on their respective buses, not independent losses
    to multiply together. effective_gbs = min(DDR*ddr_eff, AXI*axi_eff).
    Actual shared-controller arbitration can lower bandwidth further.
    """
    ddr_eff: float = 0.80
    axi_ports: int = 4
    axi_bits: int = 128
    axi_mhz: float = 200.0
    axi_eff: float = 0.85
    vpu_mhz: float = 200.0
    lanes: int = 128
    page_bytes: int = 8192
    fifo_pages: int = 8
    spu_elems: int = 4
    host_overhead_us: float = 100.0

    def __post_init__(self):
        for name in ("ddr_eff", "axi_eff"):
            v = getattr(self, name)
            if not math.isfinite(v) or not 0 < v <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("axi_mhz", "vpu_mhz"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.host_overhead_us) or self.host_overhead_us < 0:
            raise ValueError("host_overhead_us must be finite and nonnegative")
        if type(self.axi_ports) is not int or not 1 <= self.axi_ports <= 4:
            raise ValueError("axi_ports must be an integer in [1, 4] (HP0..HP3)")
        if self.axi_bits not in (32, 64, 128):
            raise ValueError("axi_bits must be 32, 64 or 128")
        for name in ("lanes", "fifo_pages", "spu_elems"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.lanes > KV260_BOARD.dsp48e2:
            raise ValueError("one multiplier per DSP lane exceeds K26 DSP capacity")
        if self.page_bytes not in (4096, 8192):
            raise ValueError("page_bytes must be 4096 or 8192")

    @property
    def axi_peak_gbs(self):
        return self.axi_ports * self.axi_bits / 8 * self.axi_mhz / 1000

    @property
    def effective_gbs(self):
        return min(KV260_BOARD.ddr_peak_gbs * self.ddr_eff,
                   self.axi_peak_gbs * self.axi_eff)

    @property
    def fifo_bytes(self):
        return self.page_bytes * self.fifo_pages

    def analytical(self):
        """Adapter retaining perf_model.Platform's actual DDR peak semantics."""
        return Platform(
            name=KV260_BOARD.name, peak_gbs=KV260_BOARD.ddr_peak_gbs,
            eff=self.effective_gbs / KV260_BOARD.ddr_peak_gbs,
            mhz=self.vpu_mhz, lanes=self.lanes, cores=1,
            token_overhead_us=self.host_overhead_us,
        )

    def event_model(self):
        from cyclesim import HW
        return HW(
            mhz=self.vpu_mhz, lanes=self.lanes, lanes_prefill=self.lanes,
            ddr_gbs=KV260_BOARD.ddr_peak_gbs,
            ddr_eff=self.effective_gbs / KV260_BOARD.ddr_peak_gbs,
            fifo_bytes=self.fifo_bytes, spu_elems=self.spu_elems,
        )
