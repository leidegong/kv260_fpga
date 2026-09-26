# KV260 block design + bitstream build for the Step 3 PL IP.
# NOT EXECUTED in this repository (no Vivado available when written); treat the
# first run as bring-up and check every reported warning.
#
#   vivado -mode batch -source step3/kv260/build_bd.tcl -tclargs bw 200
#   vivado -mode batch -source step3/kv260/build_bd.tcl -tclargs accel 200
#
# bw    : rtl/bw_test_top.sv (M1 bandwidth/correctness IP) - the first thing to build.
# accel : rtl/accel_top.sv, the functional single-cycle-FP baseline. Expect a large
#         design with negative slack at 200 MHz; it is here to exercise the flow,
#         not as a timing-closed accelerator.
#
# Requires Vivado with the Kria KV260 board files (SOM board part *kv260_som*).
# The PS preset from the board file configures DDR; HP0..HP3 (S_AXI_HPn_FPD) are
# 128-bit, M_AXI_HPM0_FPD drives the AXI-Lite control port through SmartConnect.
# Outputs in build/bd_<design>/: .xsa (with bitstream), address map, reports.

set design [expr {$argc > 0 ? [lindex $argv 0] : "bw"}]
set clk_mhz [expr {$argc > 1 ? [lindex $argv 1] : 200}]
if {$design ni {bw accel}} { error "design must be bw or accel" }
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root build bd_$design]
set part xck26-sfvc784-2LV-c

create_project kv260_$design $out -part $part -force
set bp [get_board_parts -quiet -latest_file_version {*kv260_som*}]
if {$bp eq ""} { error "KV260 SOM board part not installed (Vivado board store: kv260_som)" }
set_property board_part $bp [current_project]

set sv {kv260_regs_pkg.sv fp32_pkg.sv w4a16_dot.sv page_demux.sv bfp_quant.sv gemv_core.sv
        axil_slave.sv axi_rd_mport.sv axi_wr_stream.sv bw_test_top.sv accel_core.sv accel_top.sv}
foreach f $sv { add_files -norecurse [file join $root rtl $f] }
set_property file_type SystemVerilog [get_files *.sv]
add_files -norecurse [file join $root rtl kv260_${design}_wrapper.v]
update_compile_order -fileset sources_1

create_bd_design top
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP5 {1} CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clk_mhz \
] $ps

set core [create_bd_cell -type module -reference kv260_${design}_wrapper core]
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst]
set ic [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect ctrl_ic]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $ic

set clk [get_bd_pins ps/pl_clk0]
foreach pin {core/clk rst/slowest_sync_clk ctrl_ic/aclk ps/maxihpm0_fpd_aclk
             ps/saxihp0_fpd_aclk ps/saxihp1_fpd_aclk ps/saxihp2_fpd_aclk ps/saxihp3_fpd_aclk} {
    connect_bd_net $clk [get_bd_pins $pin]
}
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins core/rst_n]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins ctrl_ic/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins ctrl_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl_ic/M00_AXI] [get_bd_intf_pins core/s_axi_ctrl]
for {set i 0} {$i < 4} {incr i} {
    connect_bd_intf_net [get_bd_intf_pins core/m_axi_hp$i] [get_bd_intf_pins ps/S_AXI_HP${i}_FPD]
}
assign_bd_address
validate_bd_design
save_bd_design

# Address map for the host drivers (kv260/bw_driver.py --regs-phys, UIO device tree).
set fp [open [file join $out address_map.txt] w]
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces -of_objects [get_bd_cells ps]]] {
    puts $fp "[get_property NAME $seg] offset=[get_property OFFSET $seg] range=[get_property RANGE $seg]"
}
foreach space [get_bd_addr_spaces core/*] {
    foreach seg [get_bd_addr_segs -of_objects $space] {
        puts $fp "[get_property NAME $space]: [get_property NAME $seg] offset=[get_property OFFSET $seg] range=[get_property RANGE $seg]"
    }
}
close $fp

make_wrapper -files [get_files top.bd] -top
add_files -norecurse [file join $out kv260_$design.gen sources_1 bd top hdl top_wrapper.v]
set_property top top_wrapper [current_fileset]
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "implementation failed" }
open_run impl_1
report_utilization -file [file join $out utilization.rpt]
report_timing_summary -file [file join $out timing.rpt]
set wns [get_property STATS.WNS [get_runs impl_1]]
write_hw_platform -fixed -include_bit -force [file join $out kv260_$design.xsa]
set fp [open [file join $out summary.txt] w]
puts $fp "design=$design clock=${clk_mhz}MHz vivado=[version -short] WNS=$wns"
puts $fp [expr {$wns < 0 ? "TIMING NOT MET: do not use this bitstream for measurements" : "timing met"}]
close $fp
