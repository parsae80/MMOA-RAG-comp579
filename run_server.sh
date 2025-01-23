#!/bin/bash

for i in {0..7}
do
  port=$((8000 + $i))
  CUDA_VISIBLE_DEVICES=$i python flask_server.py $port > "flask_server_output_$port.txt" 2>&1 &
done