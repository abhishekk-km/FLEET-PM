# FleetPM system log

NOTE: the consolidated sections 0-14 are NOT present in this file. It holds only auto-generated extension entries and audit sections. Reconstruct sections 0-14 from the earlier consolidated log text before relying on this file.


## 15. Extensions audit, Tasks 1-3 (2026-10-27)
Status words as in section 0. Outputs inspected by the author.
- No deterioration: raw and offset test RMSE and NASA and offset bins reproduce section 6.4 and 7 exactly (VERIFIED).
- Task 1, conformal one-sided lower bounds (VERIFIED): nested training-only calibration selected one random cut per engine on all subsets; held-out coverage 0.895/0.903/0.897/0.900 at alpha 0.1 and 0.792/0.802/0.799/0.801 at alpha 0.2. Official test, conformal 90%: 0.920/0.915/0.880/0.927, all engine-bootstrap CIs contain 0.90. Conformal 80% over-covers on FD002 (0.869, CI excludes 0.80). The 0.8-quantile offset covers 0.72/0.826/0.70/0.77 and under-covers on FD003 (CI [0.61, 0.78]). Cost: mean slack 6-22 cycles; on FD002/FD004 NASA of the 90% bound is about 3.4x raw. Coverage is empirical, not guaranteed; cap-saturated engines inflate marginal coverage (HYPOTHESIS, cov_lt_cap added to measure it).
- Task 2, cost-aware replay (VERIFIED run, in-sample parameters, simulated, no anomaly signal, relative cost units): at r=100 threshold policies cost 7.7-14.8 per engine vs 100 for run-to-failure and 61-89 for the best fixed interval. Break-even r vs run-to-failure 1.6-1.9 (raw), 3.0-6.5 (offset), 4.1-9.2 (90% bound). Raw, offset and bound policies are near-reparametrisations under a swept threshold, so Task 2 does not demonstrate a benefit of the bound. Raw policy avoids 1.000/0.981/0.980/0.980 of failures at r=100.
- Task 3 clustering (VERIFIED run): k=2 silhouette 0.285/0.200/0.434/0.397, ARI 0.85/0.62/0.98/0.99 for FD001-FD004. The negative control did not come out clean (FD001 is far above the weak column-shuffle null of 0.08). Two-fault-mode reading remains a HYPOTHESIS; Gaussian null and Spearman(cluster, life) added.
- Task 3 faithfulness, FD001 only (VERIFIED run): remove-top minus remove-random RMSE -0.25/-0.37/-0.03/-0.02 for k=1,2,3,5, all CIs contain 0; keep-top vs keep-random mixed sign, CIs contain 0. No evidence top-SHAP sensors are uniquely necessary; redundancy is a HYPOTHESIS. Work-order sensor lists are attributions only.
- UNVERIFIED: faithfulness full-model CV RMSE 14.51 vs 13.85 deployed OOF (300 vs 500 trees or design() mismatch); resolve with scripts/12_explain.py --fd FD001 --skip-cluster --check.
- Defect fixed: save_merge overwrote whole subset entries, erasing FD001 clustering from ext_explain.json.

## Task1 conformal FD001 (2026-10-27 15:24 UTC)
Auto-appended from script output. Audit pending until reviewed.
- variant one_cut_per_engine; test cov a0.2 0.770, a0.1 0.920, offset 0.720, raw 0.370; empirical coverage only, no guarantee claimed

## Task1 conformal FD002 (2026-10-27 15:24 UTC)
Auto-appended from script output. Audit pending until reviewed.
- variant one_cut_per_engine; test cov a0.2 0.869, a0.1 0.915, offset 0.826, raw 0.544; empirical coverage only, no guarantee claimed

## Task1 conformal FD003 (2026-10-27 15:24 UTC)
Auto-appended from script output. Audit pending until reviewed.
- variant one_cut_per_engine; test cov a0.2 0.810, a0.1 0.880, offset 0.700, raw 0.450; empirical coverage only, no guarantee claimed

## Task1 conformal FD004 (2026-10-27 15:24 UTC)
Auto-appended from script output. Audit pending until reviewed.
- variant one_cut_per_engine; test cov a0.2 0.839, a0.1 0.927, offset 0.770, raw 0.516; empirical coverage only, no guarantee claimed

## Task3 explain FD001 (2026-10-27 15:24 UTC)
Auto-appended from script output. Audit pending until reviewed.
- cluster k=2 silhouette 0.285, ARI 0.847, null p95 0.307; mapping to fan/HPC is HYPOTHESIS

## 16. Extensions audit 2 (2026-10-28)
- RESOLVED (VERIFIED): faithfulness refit at 500 trees reproduces saved deployed OOF RMSE for fold 1 exactly (15.4576), so design() and column order match. 300 trees cost about 0.4 RMSE on fold 1 (15.84 vs 15.46). The 14.51 vs 13.85 aggregate gap is UNVERIFIED (probably fold-mean vs pooled averaging).
- Conformal coverage on engines with true RUL below the cap (VERIFIED): 90% bound 0.910/0.891/0.859/0.901; 80% bound 0.742/0.832/0.776/0.779; offset 0.8 0.685/0.777/0.647/0.685; raw 0.315/0.436/0.376/0.392 (FD001-FD004). Cap-saturated engines inflated marginal coverage by about 0.01-0.03. No CIs computed for this subset.
- Clustering with Gaussian-covariance null (VERIFIED run): k=2 silhouette 0.285/0.200/0.434/0.397 vs null p95 0.307/0.198/0.321/0.304. Negative control FD001 inside the null, FD002 borderline, FD003 and FD004 clearly above. Spearman(cluster, life) -0.18/-0.43/-0.63/-0.51: clusters track engine life, so degradation speed is a confound. Fan/HPC mapping remains a HYPOTHESIS; readme-vs-paper discrepancy unresolved.
- Work orders live in artifacts/work_orders_v2/FD00X/. Dashboard v2 searches recursively.
- Defect: docs/SYSTEM_LOG.md lacked sections 0-14 (header added, duplicate auto entries removed, log() now replaces its own section on rerun).
- Added fixed-threshold (15/30/60) cost comparison to Task 2, because swept thresholds make the three signals near-reparametrisations. Result pending.

## Task2 cost policy FD001 (2026-10-28 01:09 UTC)
Auto-generated from script output. Audit pending until reviewed.
- break-even r vs run-to-failure: rul_raw 1.6257556664437942, rul_offset 3.036771118035458, rul_conformal10 4.150404757850476
- replay on training trajectories with cross-fitted OOF bounds; simulated; no invented costs

## Task2 cost policy FD002 (2026-10-28 01:09 UTC)
Auto-generated from script output. Audit pending until reviewed.
- break-even r vs run-to-failure: rul_raw 1.8041864093920723, rul_offset 3.739937302478797, rul_conformal10 5.111433483440167
- replay on training trajectories with cross-fitted OOF bounds; simulated; no invented costs

## Task2 cost policy FD003 (2026-10-28 01:09 UTC)
Auto-generated from script output. Audit pending until reviewed.
- break-even r vs run-to-failure: rul_raw 1.6831803533309566, rul_offset 3.2550885998350583, rul_conformal10 4.448782831127584
- replay on training trajectories with cross-fitted OOF bounds; simulated; no invented costs

## Task2 cost policy FD004 (2026-10-28 01:09 UTC)
Auto-generated from script output. Audit pending until reviewed.
- break-even r vs run-to-failure: rul_raw 1.8679135990207825, rul_offset 6.517339604882424, rul_conformal10 9.221978823334327
- replay on training trajectories with cross-fitted OOF bounds; simulated; no invented costs

## Task3 explain FD003 (2026-10-28 10:25 UTC)
Auto-generated from script output. Audit pending until reviewed.
- faithfulness remove-diff by k: {'1': 0.12, '2': 0.258, '3': 0.08, '5': -0.344}

## Task3 explain FD002 (2026-10-28 10:36 UTC)
Auto-generated from script output. Audit pending until reviewed.
- faithfulness remove-diff by k: {'1': 0.451, '2': 1.051, '3': 2.05, '5': 2.03}

## Task3 explain FD004 (2026-10-28 10:44 UTC)
Auto-generated from script output. Audit pending until reviewed.
- faithfulness remove-diff by k: {'1': 0.402, '2': 0.47, '3': 0.462, '5': 1.004}
