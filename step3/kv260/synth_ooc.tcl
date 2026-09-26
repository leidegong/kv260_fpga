# Synthesize independent leaf cores for K26. No PS, AXI DMA, complete accelerator
# or bitstream is generated. Run from any working directory using Vivado.
# vivado -mode batch -source step3/kv260/synth_ooc.tcl
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root build synth_ooc]
set part xck26-sfvc784-2LV-c
if {[llength [get_parts -quiet $part]] != 1} {
    error "K26 part unavailable: install Kria/Zynq UltraScale+ device support for $part"
}
file mkdir $out
foreach top {w4a16_dot page_demux} {
    create_project -in_memory -part $part
    read_verilog -sv [file join $root rtl ${top}.sv]
    synth_design -top $top -part $part -mode out_of_context
    create_clock -name core_clk -period 5.000 [get_ports clk]
    report_utilization -file [file join $out ${top}_utilization.rpt]
    report_timing_summary -file [file join $out ${top}_synth_timing.rpt]
    write_checkpoint -force [file join $out ${top}_synth.dcp]
    close_project
}
set fp [open [file join $out scope.txt] w]
puts $fp "Vivado [version -short]; target $part; requested clock 200 MHz."
puts $fp "Leaf synthesis only. IO paths are not constrained as a board interface."
puts $fp "No place/route, timing closure, AXI subsystem, SPU/DCU RTL, or bitstream."
close $fp
