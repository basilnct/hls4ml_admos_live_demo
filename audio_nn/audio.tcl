create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

############## Add the audio codec IP and necessary IO
# First create a 10 MHz clock
# Create instance: clk_wiz_10MHz, and set properties
set clk_wiz_10MHz [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_10MHz ]
set_property -dict [ list \
 CONFIG.CLKOUT1_JITTER {290.478} \
 CONFIG.CLKOUT1_PHASE_ERROR {133.882} \
 CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {10.000} \
 CONFIG.MMCM_CLKFBOUT_MULT_F {15.625} \
 CONFIG.MMCM_CLKOUT0_DIVIDE_F {78.125} \
 CONFIG.MMCM_DIVCLK_DIVIDE {2} \
 CONFIG.RESET_PORT {resetn} \
 CONFIG.RESET_TYPE {ACTIVE_LOW} \
] $clk_wiz_10MHz
apply_bd_automation -rule xilinx.com:bd_rule:board -config { Board_Interface {sys_clock ( System Clock ) } Manual_Source {Auto}}  [get_bd_pins clk_wiz_10MHz/clk_in1]
apply_bd_automation -rule xilinx.com:bd_rule:board -config { Manual_Source {Auto}}  [get_bd_pins clk_wiz_10MHz/resetn]

# Create ports
set audio_clk_10MHz [ create_bd_port -dir O -type clk audio_clk_10MHz ]
set bclk [ create_bd_port -dir O bclk ]
set codec_addr [ create_bd_port -dir O -from 1 -to 0 codec_addr ]
set lrclk [ create_bd_port -dir O lrclk ]
set sdata_i [ create_bd_port -dir I sdata_i ]
set sdata_o [ create_bd_port -dir O sdata_o ]

# Create instance: audio_codec_ctrl_0, and set properties
set audio_codec_ctrl_0 [ create_bd_cell -type ip -vlnv xilinx.com:user:audio_codec_ctrl:1.0 audio_codec_ctrl_0 ]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/audio_codec_ctrl_0/S_AXI} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins audio_codec_ctrl_0/S_AXI]
connect_bd_net -net SDATA_I_1 [get_bd_ports sdata_i] [get_bd_pins audio_codec_ctrl_0/sdata_i]
connect_bd_net [get_bd_ports audio_clk_10MHz] [get_bd_pins clk_wiz_10MHz/clk_out1]
connect_bd_net -net audio_codec_ctrl_0_BCLK [get_bd_ports bclk] [get_bd_pins audio_codec_ctrl_0/bclk]
connect_bd_net -net audio_codec_ctrl_0_LRCLK [get_bd_ports lrclk] [get_bd_pins audio_codec_ctrl_0/lrclk]
connect_bd_net -net audio_codec_ctrl_0_SDATA_O [get_bd_ports sdata_o] [get_bd_pins audio_codec_ctrl_0/sdata_o]
connect_bd_net -net audio_codec_ctrl_0_codec_address [get_bd_ports codec_addr] [get_bd_pins audio_codec_ctrl_0/codec_address]
