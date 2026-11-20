# VGGT front end on ETH3D (stage 1, GPU)

Stage 1 runs VGGT-1B on image pairs of ETH3D sub-scenes under clean, fog, low-light and rain conditions and
writes one `.npz` per sub-scene and condition. Stage 2 (CARA and baselines) runs on a CPU; see the top-level README.
The outputs used in the paper are already in `data/eth3d_vggt` (first run) and `data/eth3d_vggt_v2`
(confirmatory run), so stage 1 is only needed to recompute them.

## Environment (Google Colab, free T4 is enough)
Use `CARA_colab_v2.ipynb` (confirmatory run) or `CARA_colab.ipynb` (first run). They install VGGT without its
pinned dependencies (`pip install --no-deps -e vggt`), download the ETH3D high-res multi-view training set
(undistorted DSLR images, about 5.5 GB) to Google Drive, and resume after disconnects.

## Commands
First run (fixed split: 6 calibration scenes, 7 test scenes):
```
python frontend_vggt.py --dataset eth3d --root <eth3d> --out out_eth3d --role calib \
  --scenes-file splits/eth3d_calib.txt --n-sub 12 --per-scene 24 --k 6 --batch 8 --conds clean
python frontend_vggt.py --dataset eth3d --root <eth3d> --out out_eth3d --role test \
  --scenes-file splits/eth3d_test.txt --n-sub 12 --per-scene 10 --k 6 --batch 8 \
  --conds clean,fog0.7,fog0.5,fog0.3,low0.3,low0.15,rain0.5
python check_frontend.py --in out_eth3d
```
Confirmatory run (all 13 scenes, leave-one-scene-out in stage 2; about 9 GPU hours on a T4):
```
python frontend_vggt.py --dataset eth3d --root <eth3d> --out out_eth3d_v2 --cache out_eth3d/_degraded \
  --role cv --seed 3 --n-sub 12 --per-scene 12 --n-test-sub 8 --k 6 --batch 8 \
  --conds clean,fog0.7,fog0.5,fog0.3,low0.3,low0.15,rain0.5
```
Fog is rendered with depth that VGGT predicts on the clean image; ETH3D's ground-truth depth maps match the
distorted images, not the undistorted ones used here. `check_frontend.py` should report a clean median edge error
of about 1 degree, a "transposed" error near 90 degrees and a negative Spearman correlation between confidence and error.
The IMC 2022 reader in `datasets.py` is kept for completeness; IMC 2022 data could not be downloaded for this study.
