#!/bin/bash
echo "Hardware topology:" > affinity_results.txt
lscpu >> affinity_results.txt

echo "Running Control baseline..." | tee -a affinity_results.txt
python speedtest.py --exp_name control --mode centralized --requires_state True --init_mode parallel_first_touch >> affinity_results.txt 2>&1

echo "Running with Thread Pinning..." | tee -a affinity_results.txt
OMP_PLACES=cores OMP_PROC_BIND=true python speedtest.py --exp_name pinned --mode centralized --requires_state True --init_mode parallel_first_touch >> affinity_results.txt 2>&1

echo "Running Serial Init (No First Touch)..." | tee -a affinity_results.txt
OMP_PLACES=cores OMP_PROC_BIND=true python speedtest.py --exp_name serial --mode centralized --requires_state True --init_mode serial >> affinity_results.txt 2>&1
