# Pre-registration for ETH3D run 2 (written 2026-09-26, before run 2 data exist)

Run 2 (`out_eth3d_v2`, seed 3, new sub-scenes, all 13 scenes, leave-one-scene-out calibration) is the
confirmatory test for the bound extensions. Everything below was fixed after looking at run 1 results.

Design history (disclosed in the paper):
- E1 (failure-mode augmentation) and E2 (learned normalizer, group-conditional thresholds) were designed on
  ETH3D *calibration* scenes with injected failures (leave-one-scene-out), see `dev_eth3d.py`, `dev_eval.py`.
- The five variants in `eval_bounds.py` were then evaluated on run 1 test scenes, which had been seen in
  aggregate before. On run 1, E2 without augmentation gave the best coverage on hard cameras.

Primary hypotheses for run 2 (one-sided, unit = test scene, exact sign-flip test, Holm over H1-H3):
- H1: E2 raises coverage over base on the hardest quarter of cameras (by u), pooled over the 6 degraded conditions.
- H2: E2 raises overall coverage over base, pooled over the 6 degraded conditions.
- H3: E2's normalizer ranks catastrophic cameras (error > 30 deg) better than u (per-scene AUROC, pooled).
Secondary: all five variants reported; width among unflagged cameras; flag rate; clean coverage.
Primary variant: E2 = Config(cam_norm="learn", cam_groups=3), alpha_cam = 0.1, calibrated on clean scenes.
No setting is changed after run 2 results are seen.

## Additional run-2 hypotheses (E3, E5), fixed before run 2 data exist
- E3 features and model: `cara/edge_uncertainty.py` (f1-f5, linear model for log error, fitted on the D_fit split).
- H4: CARA with the learned sigma has higher AUC@5 than CARA with the pre-registered sigma (f1), pooled over all 7 conditions.
- H5: outlier-edge AUROC (error > 5 deg) of the learned sigma exceeds that of f1 (per scene, pooled).
- E5 (`cara/translation.py`): LUD translation averaging on VGGT relative translations, rotated by each method's
  global rotations, all edges; CARA additionally with only its accepted edges ("CARA+gate").
- H6: CARA+gate has lower median scale-free position error than CG (all edges), pooled over conditions.
Unit of inference: test scene (13 scenes, leave-one-scene-out calibration), exact sign-flip test; Holm over H1-H6.

## Note added after the run-1 per-camera analysis (still before run 2 data exist)
On run 1, the learned normalizer did not rank catastrophic cameras better than u (AUROC 0.81-0.90 for u), so H3 is
expected to fail; E2's coverage gain comes from the group-conditional thresholds. A secondary variant E2u
(groups on u, no learned normalizer) is added to `eval_bounds.py`. H1-H6 and the primary variant are unchanged.
