#!/bin/bash

# Compile the CUDA benchmark test
nvcc -Wno-deprecated-gpu-targets -arch=sm_89 -lcublas benchmark_test.cu -o benchmark_test

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running benchmark..."
    ./benchmark_test
else
    echo "Compilation failed."
    exit 1
fi