#!/bin/bash

declare -a DELTA=("0.01" "0.02")
chosen_graph="3elt.graph"
echo $var

for val in ${DELTA[@]}; do
   var=`$inter ./run.sh -graph ./graphs/$chosen_graph -delta $val`
   echo "DELTA $val" >> loop_output.txt
   echo "$var" >> loop_output.txt
done
