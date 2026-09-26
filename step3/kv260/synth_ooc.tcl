# Out-of-context synthesis of the Step 3 RTL for K26. No PS, no bitstream.
#   vivado -mode batch -source step3/kv260/synth_ooc.tcl [-tclargs top1 top2 ...]
# Default tops: the leaf cores and the M1 bandwidth IP. accel_top is the functional
# single-cycle-FP baseline and is expected to be large and far from 200 MHz;
# synthesize it explicitly (-tclargs accel_top) to get its utilization picture.
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root build synth_ooc]
set part xck26-sfvc784-2LV-c
if {[llength [get_parts -quiet $part]] != 1} {
    error "K26 part unavailable: install Kria/Zynq UltraScale+ device support for $part"
}
set sources {
    kv260_regs_pkg.sv fp32_pkg.sv w4a16_dot.sv page_demux.sv bfp_quant.sv gemv_core.sv
    axil_slave.sv axi_rd_mport.sv axi_wr_stream.sv bw_test_top.sv accel_core.sv accel_top.sv
}
set tops [expr {$argc > 0 ? $argv : {w4a16_dot page_demux bfp_quant gemv_core axi_rd_mport axi_wr_stream bw_test_top}}]
file mkdir $out
set fp [open [file join $out scope.txt] w]
puts $fp "Vivado [version -short]; target $part; requested clock 200 MHz; tops: $tops"
puts $fp "Out-of-context synthesis only: no place/route, no timing closure, no bitstream."
foreach top $tops {
    create_project -in_memory -part $part
    foreach f $sources { read_verilog -sv [file join $root rtl $f] }
    synth_design -top $top -part $part -mode out_of_context
    create_clock -name core_clk -period 5.000 [get_ports clk]
    report_utilization -file [file join $out ${top}_utilization.rpt]
    report_timing_summary -file [file join $out ${top}_synth_timing.rpt]
    write_checkpoint -force [file join $out ${top}_synth.dcp]
    puts $fp "$top: see ${top}_utilization.rpt / ${top}_synth_timing.rpt"
    close_project
}
close $fp
