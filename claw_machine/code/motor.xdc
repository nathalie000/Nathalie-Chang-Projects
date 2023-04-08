set_property IOSTANDARD LVCMOS33 [get_ports {left[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {left[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {right[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {right[0]}]
set_property PACKAGE_PIN C15 [get_ports left_motor]
set_property PACKAGE_PIN B15 [get_ports right_motor]
set_property PACKAGE_PIN A15 [get_ports {left[1]}]
set_property PACKAGE_PIN A17 [get_ports {left[0]}]
set_property PACKAGE_PIN A14 [get_ports {right[1]}]
set_property PACKAGE_PIN A16 [get_ports {right[0]}]
set_property PACKAGE_PIN R2 [get_ports switch1]
set_property PACKAGE_PIN T1 [get_ports switch2]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports left_motor]
set_property IOSTANDARD LVCMOS33 [get_ports right_motor]
set_property IOSTANDARD LVCMOS33 [get_ports switch1]
set_property IOSTANDARD LVCMOS33 [get_ports switch2]
set_property IOSTANDARD LVCMOS33 [get_ports front_motor]
set_property IOSTANDARD LVCMOS33 [get_ports {front[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {front[0]}]
set_property PACKAGE_PIN L17 [get_ports {front[1]}]
set_property PACKAGE_PIN M19 [get_ports {front[0]}]
set_property PACKAGE_PIN P17 [get_ports front_motor]

set_property PACKAGE_PIN T17 [get_ports reset]
set_property IOSTANDARD LVCMOS33 [get_ports reset]

set_property IOSTANDARD LVCMOS33 [get_ports {LED[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {LED[2]}]
set_property PACKAGE_PIN V19 [get_ports {LED[3]}]
set_property PACKAGE_PIN U19 [get_ports {LED[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {LED[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {LED[0]}]
set_property PACKAGE_PIN E19 [get_ports {LED[1]}]
set_property PACKAGE_PIN U16 [get_ports {LED[0]}]

set_property PACKAGE_PIN T18 [get_ports push]
set_property IOSTANDARD LVCMOS33 [get_ports push]

set_property IOSTANDARD LVCMOS33 [get_ports {AN[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {AN[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {AN[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {AN[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {display[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {display[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {display[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {display[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {display[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {display[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {display[0]}]
set_property PACKAGE_PIN W4 [get_ports {AN[3]}]
set_property PACKAGE_PIN V4 [get_ports {AN[2]}]
set_property PACKAGE_PIN U4 [get_ports {AN[1]}]
set_property PACKAGE_PIN U2 [get_ports {AN[0]}]
set_property PACKAGE_PIN U7 [get_ports {display[6]}]
set_property PACKAGE_PIN V5 [get_ports {display[5]}]
set_property PACKAGE_PIN U5 [get_ports {display[4]}]
set_property PACKAGE_PIN V8 [get_ports {display[3]}]
set_property PACKAGE_PIN U8 [get_ports {display[2]}]
set_property PACKAGE_PIN W6 [get_ports {display[1]}]
set_property PACKAGE_PIN W7 [get_ports {display[0]}]

set_property PACKAGE_PIN C17 [get_ports PS2_CLK]						
	set_property IOSTANDARD LVCMOS33 [get_ports PS2_CLK]
	set_property PULLUP true [get_ports PS2_CLK]
set_property PACKAGE_PIN B17 [get_ports PS2_DATA]					
	set_property IOSTANDARD LVCMOS33 [get_ports PS2_DATA]	
	set_property PULLUP true [get_ports PS2_DATA]
set_property PACKAGE_PIN W5 [get_ports clk]							
	set_property IOSTANDARD LVCMOS33 [get_ports clk]
	create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]	
