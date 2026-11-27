# H100 Waffle Pi Gazebo campaign handoff

## Scope and current decision

All work described here ran on the H100 server at `/home/rl-group4/Projects/Untapped/RL_Demo`; the experiment root is `/home/rl-group4/waffle_runs/RL_Demo-20261126T161914Z`. Runs used the official TurtleBot3 Waffle Pi SDF in a rootless Gazebo Harmonic container and actual DR-SPAAM inference on simulated scans. They did not contact or command a physical robot.

The campaign did **not** qualify a policy. The selected seed 434 checkpoint 00040 achieved 8/9 validation successes, then 4/6 successes on its one allowed six-layout held-out test (one collision and one timeout). The six test layouts are consumed. Do not retune on them or rerun them. The profile is illustrative: there is no measured Waffle calibration record. No measured-fidelity, hardware-safety, or deployment claim follows from these results.

## Reproducible source and run evidence

The checkout started at `e5a6da0a938a581d91e6ac7f266bb7c7bf105611` on `main`; the mission work is on `codex/waffle-gazebo-mission-20261126`. Mission scripts, logs, model and detector provenance, training checkpoints, split manifests, and JSON metrics are under the experiment root's `mission/`, `logs/`, `manifests/`, `models/`, and `experiments/` directories. `mission/state.json` is the machine-readable ledger; `mission/STATUS.md` records chronological evidence and `mission/ISSUES.md` records open limitations.

The actual held-out result is `experiments/eval_v3rep_seed434_cp40_test_all6/metrics.json`. The selected checkpoint and immutable illustrative profile hashes are recorded in `mission/state.json`. Do not treat the old held-out SD as valid: the previous evaluator used future-window positions for SD, and no per-timestep trace was saved to recompute it. The test outcome counts and per-episode ITR observations remain historical evidence; the test is still consumed.

The corrected metric path was exercised in real Gazebo on three validation scenarios using checkpoint 00040. Evidence is `experiments/eval_v3rep_seed434_cp40_itrfix_validation_screen3/metrics.json` and its stdout log. It completed two successes and one collision. Mean per-episode ITR was 0.047619; event-pooled current-time SD was 0.880662 m across four intrusion timesteps. This is a metric-path integration check, not a policy qualification. The earlier `metricfix` check's aggregate ITR was incorrect; use the `itrfix` output.

## Metric definitions

These definitions follow the cited paper text stored at `references/paper.txt`, lines 305-319:

- For each episode, an intrusion timestep is one where the robot center is within the sum of robot and human radii of any ground-truth human position from t+1 through t+5.
- Episode ITR is the number of intrusion timesteps divided by that episode's policy-step count. Reported ITR is the unweighted mean of per-episode ratios, matching the paper.
- Episode SD is the mean current-time center-to-center distance to the closest human, evaluated at intrusion timesteps. Reported SD pools those observations across episodes and divides by the total number of intrusion timesteps. SD is undefined when there are no intrusion observations.
- Gazebo completes the final lookahead by advancing simulated humans four additional policy frames after termination. Robot state remains at its terminal position; no policy action is taken during this suffix. These frames are metric lookahead only and are not counted as policy steps.

## Validation performed after the metric correction

`bash scripts/validate_waffle.sh` passed with 67 tests and one optional RVO2 test skipped. The mission-local dependency paths are needed on this server; no shared FedQCL environment was changed. `git diff --check` passed. The focused paper-metric tests cover future-truth ITR, current-time SD, complete horizons, strict contact boundary, undefined SD, and unequal episode/event aggregation.

This test run does not prove native physics qualification, detector accuracy, broad policy generalization, physical fidelity, or training convergence. The pre-existing native Gazebo and detector integration smokes are recorded in `mission/STATUS.md`; the native GitHub Actions workflow and full-scale benchmark suites have not been demonstrated by this handoff.

## Continue safely

1. Preserve the consumed split and all existing mission artifacts.
2. Obtain measured robot geometry, motor response, LiDAR extrinsics/range, safety stopping and gate calibration before claiming measured fidelity or moving beyond simulation-only work.
3. Expand the scenario bank and establish a frozen validation-only model-selection rule before generating a new independent test set.
4. Use corrected metrics, record per-timestep trace provenance when permitted, and never select or tune from a final test set.
5. Run native Gazebo/scan parity, scale, throughput, and complete training/evaluation gates before making a deployment recommendation. A smoke run is not qualification.

The physical robot remains uncommanded. Any physical calibration or operation requires a separate, explicit authorization and safety process.
