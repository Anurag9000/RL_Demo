# Seed437 update340 sigmoid-forward counterfactual

## Protocol

- H100 GPU0, run 2026-11-27T16:27:35Z through 2026-11-27T16:32:53Z (exit 0, 307.0s).
- Same seed437 update340 checkpoint `889dd923daf4356fc4c7627aa6138c371366243c6130f60061517343b50ebe8a` and same 100 unique validation scenarios as the checkpoint-native diagnostic; seed 900440, eight Gazebo lanes. The test split was not read.
- Counterfactual mapping was `v=max_linear*sigmoid(z0)`, `w=max_angular*tanh(z1)`. The existing safety gate and simulator remained unchanged. This is a diagnostic-only action-transform override; it is not a valid checkpoint contract or qualified policy.
- Captured 609 policy-step action records, 3,045 guard substeps, 100 compressed traces, and 116 worker log files. Offline paper-trace verification passed.

## Matched result

| Measure | Checkpoint-native map | Sigmoid counterfactual |
|---|---:|---:|
| Successes | 0/100 | 0/100 |
| Collisions | 100/100 | 100/100 |
| Timeouts | 0/100 | 0/100 |
| Mean path | 0.001010 m | 0.021136 m |
| Mean duration | 1.5125 s | 1.5225 s |
| Gate intervention fraction | 65.455% | 65.353% |
| Episodes with gate activity | 95% | 97% |

The scenarios match one-to-one. Mean paired path change was 0.020127 m. Under the counterfactual, requested forward velocity averaged 0.03773 m/s (range 0.03581 to 0.04222); the positive mapping restored some motion, but all episodes still collided. The gate blocked 1751/3045 substeps (57.504%); sensor/tracker faults occurred in 85/3045 substeps (2.791%).

## Interpretation

The forward-action projection explains the near-zero path under the checkpoint-native mapping, but changing that map alone does not fix navigation: the matched sigmoid counterfactual still produced no successes and collisions in all 100 cases, while the gate was active in 97% of episodes. That leaves the controller's action choice, close human interactions, and gate deadlock behavior to investigate together. Keep the safety gate intact; do not claim improvement from the small path increase. This is a single-checkpoint, 100-case diagnostic, not a full-bank qualification.
