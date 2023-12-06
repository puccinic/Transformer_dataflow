open_project -reset Transformer_accel
set_top accel
add_files Accel.cpp -cflags "-ILayers"
add_files -tb TestBenchAccel.cpp -cflags "-ITests -Wno-unknown-pragmas"
add_files -tb input1.txt
add_files -tb input2.txt
add_files -tb input3.txt
add_files -tb input4.txt
add_files -tb input5.txt
add_files -tb input6.txt
add_files -tb input7.txt
add_files -tb input8.txt
add_files -tb input9.txt
add_files -tb input10.txt
add_files -tb input11.txt
add_files -tb input12.txt
add_files -tb golden_result.txt
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
