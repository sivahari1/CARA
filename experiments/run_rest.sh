#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd "$(dirname "$0")"
while kill -0 972 2>/dev/null || kill -0 973 2>/dev/null; do sleep 20; done
python3 real_notredame.py 5 > ../results/real_nd.log 2>&1
python3 extra.py ablation > ../results/ablation.log 2>&1
python3 extra.py dop > ../results/dop.log 2>&1
python3 extra.py grid > ../results/grid.log 2>&1
python3 extra.py runtime > ../results/runtime.log 2>&1
echo ALLDONE > ../results/rest.done
