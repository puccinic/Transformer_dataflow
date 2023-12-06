open_project -reset Transformer_accel
set_top accel
add_files Accel.cpp -cflags "-ILayers"
add_files -tb TestBenchAccel.cpp -cflags "-ITests -Wno-unknown-pragmas"
open_solution -reset "solution1" -flow_target vivado
set_part {xck24-ubva530-2LV-c}
create_clock -period 5 -name default
set_clock_uncertainty 2
set_directive_top -name accel "accel"
csim_design
set hls_exec 1
if {$hls_exec == 2} {
    csynth_design
    cosim_design
    export_design -format ip_catalog    
}

exit
