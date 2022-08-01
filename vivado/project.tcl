create_project pynq_z2_audio pynq_z2_audio -part xc7z020clg400-1 -force

set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
set_property  ip_repo_paths  ../../ip [current_project]
update_ip_catalog
create_bd_design "design_1"

source design.tcl
save_bd_design

make_wrapper -files [get_files pynq_z2_audio/pynq_z2_audio.srcs/sources_1/bd/design_1/design_1.bd] -top

add_files -norecurse ./pynq_z2_audio/pynq_z2_audio.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v

add_files -fileset constrs_1 -norecurse ./pins.tcl
set_property used_in_synthesis false [get_files ./pins.tcl]
set_property used_in_simulation false [get_files ./pins.tcl]

reset_run impl_1
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 6
wait_on_run -timeout 360 impl_1

open_run impl_1
report_utilization -file util.rpt -hierarchical -hierarchical_percentages

# move and rename bitstream to final location
file copy -force ./pynq_z2_audio/pynq_z2_audio.runs/impl_1/design_1_wrapper.bit audio.bit

# copy hwh files
file copy -force ./pynq_z2_audio/pynq_z2_audio.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh audio.hwh
