#!/bin/bash
echo "Hardware topology:"
lscpu

echo "Running Control baseline..."
python speedtest.py --mode centralized --requires_state True --init_mode parallel_first_touch

echo "Running with Thread Pinning..."
OMP_PLACES=cores OMP_PROC_BIND=true python speedtest.py --mode centralized --requires_state True --init_mode parallel_first_touch
