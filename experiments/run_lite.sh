#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd "$(dirname "$0")"
while [ ! -f ../results/rest.done ]; do sleep 20; done
python3 real_lite.py > ../results/real_lite.log 2>&1
echo done > ../results/lite.done
