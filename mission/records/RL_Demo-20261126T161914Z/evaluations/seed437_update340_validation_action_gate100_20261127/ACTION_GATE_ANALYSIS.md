# Seed437 update340 validation action/gate diagnostic

## Protocol and provenance

- H100 `gpu2`, GPU0; diagnostic started 2026-11-27T16:03:38Z and exited 2026-11-27T16:09:01Z with status 0 (elapsed 311.8s).
- Checkpoint `00340.pt`, SHA-256 `889dd923daf4356fc4c7627aa6138c371366243c6130f60061517343b50ebe8a`; v5 illustrative profile SHA-256 `84557d48d8b92f589ccf62e2b719510ccfdaa83a27c970d42659d4cf94b494e7`.
- Validation only: 100 episodes over 100 unique scenarios from the 500-case v5 validation split, seed 900440, eight Gazebo lanes. No test cases read.
- Captured all 605 policy-step action records, all 3,025 guard substeps, 100 compressed paper-metric traces, 116 Gazebo worker log files, runner receipt, and launcher log. Offline paper-trace metric verification passed.

## Outcome

- 0/100 successes, 100/100 collisions, 0 timeouts; mean reward -24.694.
- Mean path 0.001010 m, mean episode duration 1.5125 s, mean final goal distance 1.2279 m.
- Gate intervention fraction 65.455% across policy transitions; at least one gate intervention in 95% of episodes.
- All 100 episode minimum human clearances were negative; paper-trace offline verification passed for 100/100.

## Action and gate evidence

- `waffle_pi/evaluate.py` selects `model.act(..., deterministic=True)` for this evaluator. All 605 deterministic latent forward-action outputs were negative: mean -0.502817, median -0.512531, range [-0.583763, -0.313935].
- Under `waffle_pi.native.native_twist`, forward velocity is `max_linear * max(0, tanh(z0))`. Therefore all 3,025 physical-substep requests had exactly zero forward velocity; the requests were angular-only. This directly explains the near-zero translation for this checkpoint and sample.
- The independent safety gate blocked 1702/3,025 substeps (56.264%). Sensor/tracker fault was recorded on 85/3,025 (2.810%); 2,500 substeps tracked four humans, 440 tracked three, and 85 tracked zero. There were no substeps with zero detector points or scan age over one second.
- Initial scenario surface clearance (robot radius included) had min/p05/median/mean/max 0.0813/0.1037/0.3333/0.4366/1.2292 m. 43 cases started above 0.4 m clearance; all still collided and averaged 0.001358 m path (scenario-derived subset).

## Interpretation and limits

The primary observed failure for this checkpoint is a collapsed forward-action output: every sampled latent forward value projects to zero speed. Gate intervention also occurs frequently, but it is not sufficient to explain the stationary behavior because many guard substeps are unblocked and the policy requests no forward speed in every substep. Sensor/tracker faults are present in 2.8% of substeps and cannot explain the all-episode result by themselves. Do not loosen the safety gate to force movement; inspect training reward/advantage/action-distribution behavior and the native forward-only action contract while retaining safety checks. This result covers one checkpoint and a 100-case validation sample; it does not establish behavior on the full bank, other checkpoints, or stochastic PPO rollouts during training.
