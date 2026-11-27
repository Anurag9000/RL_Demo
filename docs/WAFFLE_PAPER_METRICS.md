# Gazebo Waffle Pi paper metric schema

This document defines the paper-style fields emitted by the Gazebo evaluator. The method paper defines SR as navigation success percentage, NT and PL over successful episodes, ITR as the mean episode ratio c/C using the robot's current position against humans' ground-truth positions from the next five timesteps, and SD as the nearest-human distance at intrusion times. See [Liu et al., ICRA 2023, Section IV-B.2](https://arxiv.org/pdf/2203.01821) for the original protocol.

## Result fields and units

| Field | Unit and aggregation |
|---|---|
| paper_sr_percent | Successes divided by all completed scheduled cases, in percent. |
| paper_cr_percent | Collisions divided by all completed scheduled cases, in percent. |
| paper_tr_percent | Timeouts divided by all completed scheduled cases, in percent. The three paper outcome rates partition the completed cases. |
| paper_nt_success_only_s | Mean simulated navigation time over successful episodes only, in seconds; null when there are no successes. |
| paper_pl_success_only_m | Mean Gazebo truth path length over successful episodes only, in meters; null when there are no successes. |
| mean_paper_itr_percent | Unweighted mean across completed episodes of each episode's c/C, in percent. Each policy step contributes at most one intrusion even if multiple people or future steps overlap. |
| mean_paper_sd_m | Pooled mean current-time center distance to the closest human over intrusion timesteps, in meters; null when there are no intrusion events. |

The existing success_rate, collision_rate, timeout_rate and mean_paper_itr values remain fractions for compatibility. mean_time_s and mean_path_m also remain available as all-outcome descriptive means; they are not paper NT/PL. The explicit paper fields above remove those unit and denominator ambiguities.

Gazebo writes a truth trace per completed episode. Ground-truth futures are used for offline evaluation only and never enter policy observations. The evaluator independently scores saved traces and rejects incomplete future horizons, duplicate or missing scenario IDs, and disagreement with online metrics. Run python -m waffle_pi.offline_metrics --trace-dir OUT/metrics_traces --output OUT/offline_metrics.json to reproduce the trace aggregation.

These metrics reproduce the paper definitions but do not make an illustrative Gazebo profile comparable to the paper's original CrowdSim results or establish physical safety. Report the robot model, scene regime, profile, case count, and uncertainty beside the metrics.
