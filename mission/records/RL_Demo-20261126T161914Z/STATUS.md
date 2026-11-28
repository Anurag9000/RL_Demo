# Waffle Pi Gazebo mission status

- Started (UTC): 20261126T161914Z
- Host: H100 server; repository: /home/rl-group4/Projects/Untapped/RL_Demo
- Phase: H100 PPO training and one-time held-out evaluation complete; policy not behaviorally qualified. All results remain simulation-only on an illustrative profile.
- Source at mission start: branch main, commit e5a6da0a938a581d91e6ac7f266bb7c7bf105611; initially clean
- Latest sync check confirmed HEAD and `origin/main` both remain e5a6da0a938a581d91e6ac7f266bb7c7bf105611. The mission branch has uncommitted source changes.
- Simulator: host `gz` remains absent; rootless image `localhost/waffle-gazebo-harmonic:mission-20261126` uses Gazebo Harmonic 8.15.0. Repository worker smoke passed 1.0 simulated second, five scans, 360 rays, floor contact observed, no contact. A stage-logged CPU smoke passed a GazeboVecEnv reset, actual DR-SPAAM scan inference, released GST predictor load, randomly initialized PPO policy action, and one Gazebo physics transition. A separate stationary zero-action run completed one 360-transition held-out validation episode in real Gazebo, generated the required four human-only lookahead transitions, and reported paper ITR=0.0 with no intrusion (paper SD undefined). Both are integration checks on the illustrative profile; they are not policy-performance or training evidence. Logs: `../logs/policy-vector-smoke-np126.stdout` and `../logs/paper-metric-smoke.stdout`.
- Python: system Python 3.12.3; system Torch, Gymnasium and DR-SPAAM are absent
- CUDA toolkit: nvcc not found; Nsight Systems present
- Compute observation: both H100s had substantial GPU memory allocations; GPU 0 was 100% utilized and included another user's large process. Existing rl-group4 FedQCL processes also occupy GPUs and CPU. No process was stopped or altered.
- CPU/memory: 64 allowed CPUs, 2 NUMA nodes; approximately 397 GiB host memory reported available, but node free memory was low and existing jobs are active.
- Detector: official RWTH/VisualComputingInstitute annotation-trained DR-SPAAM checkpoint acquired from the linked author weights folder; SHA-256 and source commit are in `../manifests/dr_spaam_provenance.json`. CPU load/forward smoke passed on a 360-point constant-range fixture; this is not accuracy or sensor evidence.
- Code changes on mission branch: paper future-truth ITR/SD calculation in Gazebo evaluation, stable scene IDs, and NumPy alias compatibility for the official detector. Paper-metric tests: 3 passed. Gazebo-vector tests: 3 passed with FakeWorker only, not Gazebo physics.
- Isolated test dependencies and DR-SPAAM runtime dependencies are under this run root; the pre-existing FedQCL environment was not modified.
- Current gates: broker allocation before training; measured Waffle Pi calibration; trained checkpoint; multi-scenario held-out evaluation and qualification. Latest broker check assigned both GPUs to other users. The profile under `profiles/` remains illustrative and cannot support a measured-fidelity claim.

No credentials are stored here. Source changes are uncommitted on the mission branch.

## Current GPU pilot (2026-11-26 18:12 UTC)

- User explicitly authorized using remaining H100 VRAM and said not to gate on utilization. Two independent PPO pilots are running, one on each H100, each with PyTorch allocator capped at 10% of total VRAM; all other processes are untouched.
- Run: `waffle_illustrative_gpu0_ppo_10k_seed425_20261126`; 10,000 transition budget, 8 Gazebo environments, 4 humans, 30-step recurrent PPO rollouts, seed 425. Profile status is illustrative.
- At last snapshot: 2,400 transitions completed; mission PID 2569474 held 2,830 MiB; GPU0 had 10,160 MiB free and GPU1 29,146 MiB free.
- Last 100 training episodes averaged -20.2 reward (collision penalty). This is an early negative pilot result on a single illustrative training scenario, not a policy-performance claim. Log: `../logs/waffle_illustrative_gpu0_ppo_10k_seed425.stdout`.
- The run writes checkpoints every 10 updates. Current PPO checkpoint state is still in progress.

- Latest two-run snapshot (2026-11-26T18:23:05Z): seed425 GPU0 4080/9840 transitions, process 2852 MiB, free 50174 MiB; seed426 GPU1 0/9840 transitions, process 2620 MiB, free 23457 MiB. Both use the same illustrative four-human train scenario; the seed425 recent reward is still -20.2.

## Latest H100 progress (2026-11-26 18:29 UTC)

- GPU0 free VRAM 49,266 MiB; GPU1 free VRAM 19,975 MiB. Mission jobs remain individually allocator-capped at 10% of device capacity; unrelated workloads remain untouched.
- Seed425 (GPU0) reached update 20 / 5,040 of 9,840 planned transitions; seed426 (GPU1) reached update 3 / 9,840. Both report recent mean/median episode reward -20.2/-20.2 (collision penalty), with slow Gazebo/PPO progress. They are still running.
- Evaluated checkpoint 00010 from seed425 on one validation scenario, `illustrative-4-validation`, using actual Gazebo and the configured DR-SPAAM + safety gate. Result: collision, 0 success, 1.0 intervention fraction, 49 counted transitions, 12.25 simulated seconds, intrusion 5.0 s, paper ITR 0.10204, paper SD 0.48838 m. Single-scenario exploratory result only; no qualification claim. Full metrics: `../experiments/eval_seed425_checkpoint10_validation/metrics.json`.
- A second one-scenario validation is running against checkpoint 00020 with the same deterministic seed and held-out scenario. Output: `../experiments/eval_seed425_checkpoint20_validation/metrics.json`.
- The first checkpoint evaluation is complete. No physical robot was contacted or commanded.

## Gate starvation fix and corrected pilot (2026-11-26 18:44 UTC)

- A real Gazebo guard diagnostic explained the earlier failure. In `illustrative-4-train`, two forward lidar octants had only 8.9% and 13.3% in-range returns (the arena wall was beyond the 3.5 m scan range), below the gate's 50% coverage check. The nearest return was 0.403 m. Both a forward command and a turn were rejected. This—not GPU utilization—made the original training actions immobile; those two runs were stopped at 6,240 and 1,200 transitions with their checkpoints and logs retained.
- A separate mission-only profile, `mission/profiles/waffle_gazebo_illustrative_safe_clearance_v1.json`, has three 4-human scenarios each for train, validation and test. Robot starts are nearer arena center, and human lanes are spaced outside the initial conservative gate envelope. On real Gazebo startup, all octants had at least 71.1% in-range returns, nearest return was 1.202 m, and both 0.1 m/s forward and 0.5 rad/s turn commands passed the gate. The original profile remains unchanged. Both profiles are illustrative; no physical Waffle calibration evidence exists.
- Corrected-profile PPO runs: seed427/GPU0 at 3,600/9,840 transitions and seed428/GPU1 at 3,840/9,840. Recent episode medians are about 10.6 and 10.7; means are lower (-6.8 and -0.7) because collision-penalty episodes still occur. Each is capped at 10% device VRAM and unrelated jobs remain untouched.
- The initial, untrained native-action checkpoint from seed427 completed all three held-out validation scenarios: 3/3 reached goal, 0 collisions, mean reward 10.41, mean SPL 1.0, gate intervention on 13.3% of steps. Paper ITR was 0 in all three and paper SD is undefined (no paper intrusion steps). Safety discomfort-time averaged 3.58 s/episode; this metric is distinct from paper ITR. Exact 3-scenario Wilson 95% success interval is 0.439-1.0, so this is only a small illustrative smoke evaluation, not qualification or evidence of learning. Full metrics: `../experiments/eval_safe_clearance_seed427_initial_all3/metrics.json`.
- A matching three-scenario held-out evaluation of trained seed427 checkpoint 00010 is running to compare with the initial checkpoint under the same split and seed.

## PPO checkpoint comparison (2026-11-26 18:47 UTC)

Using the same three held-out safe-clearance scenarios, seed427 checkpoint 00010 scored 1/3 success and 2/3 collisions (mean reward -13.41, paper ITR 0.0237, paper SD 0.518 m). Checkpoint 00020 scored 0/3 success and 3/3 collisions (mean reward -31.94, paper ITR 0.0490, paper SD 0.430 m). Both are worse than the untrained forward-biased initialization's 3/3 successes. PPO is currently degrading this simple initial controller; the result is an early negative optimization finding, not a trained policy.

- Seed427/GPU0 is at 6,000/9,840 transitions; seed428/GPU1 is at 6,480/9,840. Recent training rewards remain mixed and median is dropping for seed427.
- A three-scenario evaluation of seed428 checkpoint 00020 is underway. Full per-episode records remain under the mission `experiments/` directory.

## Second seed validation (2026-11-26 18:52 UTC)

Seed428 checkpoint 00020 passed all three held-out safe-clearance scenarios (3/3 goal reached, 0 collisions, mean reward 11.01, mean SPL 1.0, mean gate-intervention fraction 12.3%, paper ITR 0 and SD undefined). Its Wilson 95% success interval is 0.439-1.0. Seed427 checkpoint 00020 failed 3/3 scenarios. This is substantial seed variance on a small illustrative suite; retain the checkpoint as a candidate for further evaluation, not a qualified policy. Metrics: `../experiments/eval_safe_clearance_seed428_update20_all3/metrics.json`.

- Seed427/GPU0: 8,880/9,840 transitions, recent mean/median reward -9.2/-6.3.
- Seed428/GPU1: 9,120/9,840 transitions, recent mean/median reward -5.9/10.5.

## Completed 10k training runs and frozen validation (2026-11-26 18:57 UTC)

Both corrected-profile PPO runs completed 41 updates (9,840 actual transitions each). Their frozen checkpoint 00040 results on the three validation scenarios agreed: seed427 had 3/3 goal reaches and seed428 had 3/3, with no collisions. Mean rewards were 11.29 and 11.39; both mean SPL=1.0, paper ITR=0 and paper SD undefined. Mean gate intervention fractions were 10.5% and 11.7%. Each three-episode Wilson success interval is 0.439-1.0. Per-episode evidence: `../experiments/eval_safe_clearance_seed427_final40_all3/metrics.json` and `../experiments/eval_safe_clearance_seed428_final40_all3/metrics.json`.

The checkpoints are frozen. Separate 3-scenario test-split evaluations have started for both seeds. These results remain limited to an illustrative mission-only scene profile; they do not establish hardware fidelity or broad navigation performance.

## Frozen test-split results (2026-11-26 18:58 UTC)

Both final 00040 checkpoints passed all three held-out test scenarios: seed427 3/3, seed428 3/3, with zero collisions. Mean rewards were 10.52 and 9.87; SPL was 1.0 for all six episodes; paper ITR was 0 for every episode and paper SD is undefined because there were no paper intrusion steps. Gate intervention fractions were 12.5% and 15.5%. Safety discomfort-threshold time averaged 3.75 s per episode in each seed; this is a separate, looser clearance measure and does not mean collision or paper intrusion. The Wilson 95% success interval for each seed's three episodes is 0.439-1.0. Detailed results: `../experiments/eval_safe_clearance_seed427_final40_test_all3/metrics.json` and `../experiments/eval_safe_clearance_seed428_final40_test_all3/metrics.json`.

The train, validation and test profiles are small, mission-created illustrative layouts with separated pedestrian lanes. These six test episodes do not establish dense-crowd performance, physical realism, or hardware readiness. Physical Waffle calibration remains unavailable; no physical robot motion was commanded.

## Held-out test pass and added replications (2026-11-26 19:01 UTC)

Both frozen 00040 checkpoints passed all three test scenarios (seed427 and seed428 each 3/3, zero collisions, mean SPL 1.0, paper ITR 0, paper SD undefined). Test metrics are saved in the two `eval_safe_clearance_seed*_final40_test_all3/metrics.json` files. To expand the small seed sample, two new independent 10k PPO replications have started: seed429 on GPU0 and seed430 on GPU1, with the same 10% per-process VRAM cap and same frozen illustrative profile. Their logs/checkpoints will be stored separately.

Replication snapshot (2026-11-26 19:03 UTC): seed429/GPU0 reached 1,440/9,840 transitions (recent mean/median reward -4.1/11.4); seed430/GPU1 reached 960/9,840 (-1.4/11.2). Both remain active with ample free VRAM (50,976 MiB on GPU0, 29,387 MiB on GPU1).

Replication progress (2026-11-26 19:05 UTC): seed429/GPU0 reached 3,120/9,840 transitions (recent mean/median reward -5.8/10.2); seed430/GPU1 reached 2,640/9,840 (1.5/10.9). Free VRAM was 50,976 MiB and 29,347 MiB respectively. The blocker ledger now reflects that final validation/test sweeps are complete for seeds 427 and 428, while seeds 429 and 430 remain active and scenario diversity/calibration limits remain open.

Replication progress (2026-11-26 19:12 UTC): seed429/GPU0 reached 7,920/9,840 transitions (recent mean/median reward -8.7/9.6); seed430/GPU1 reached 6,960/9,840 (-10.4/8.7). Both processes remain alive and are progressing. Free VRAM: GPU0 50,976 MiB; GPU1 29,327 MiB. Recent reward means remain negative with mixed medians, so final held-out checkpoint evaluations will decide outcomes.

Replication progress (2026-11-26 19:12:45 UTC): seed429/GPU0 reached 8,640/9,840 transitions (recent mean/median reward -8.4/9.6); seed430/GPU1 reached 7,680/9,840 (-10.6/8.4). Both processes remain active. Free VRAM: GPU0 50,976 MiB; GPU1 29,305 MiB.

Replication progress (2026-11-26 19:13:53 UTC): seed429/GPU0 reached 9,360/9,840 transitions (recent mean/median reward -8.6/9.6); seed430/GPU1 reached 8,160/9,840 (-9.6/9.6). Both remain active; seed429 is near completion.

Seed429 replication training completed at checkpoint 00040 (9,840 transitions); checkpoint SHA256 `8307eca685264813d5dcfe6bc1eb38914658f201b6fa986a223660753b23abe8`. A three-scenario test-split evaluation is now running. Seed430 remains active at 8,640/9,840 transitions (recent mean/median reward -8.5/10.3). Evaluation log: `../logs/eval_safe_clearance_seed429_final40_test_all3.stdout`.

Replication progress (2026-11-26 19:16:10 UTC): seed430 reached 9,600/9,840 transitions (recent mean/median reward -8.7/10.5). Seed429 checkpoint test evaluation remains active during Gazebo startup/inference; no evaluation metrics have been written yet. Free VRAM: GPU0 53,143 MiB; GPU1 28,230 MiB.

## Expanded replication outcome and seed-generalization failure (2026-11-26 19:17 UTC)

Seed429 finished 9,840 transitions at checkpoint 00040 (SHA256 `8307eca685264813d5dcfe6bc1eb38914658f201b6fa986a223660753b23abe8`). Its final test-split result was 0/3 successes and 3/3 collisions, mean reward -27.11, SPL 0, paper ITR 0.03135 and paper SD 0.5181 m; gate intervention fraction 17.9%. This contradicts its earlier 3/3 validation pass and is direct evidence of poor generalization/seed instability on the small illustrative layout set. Metrics: `../experiments/eval_safe_clearance_seed429_final40_test_all3/metrics.json`.

Seed430 completed 9,840 transitions at checkpoint 00040 (SHA256 `a7aa03370f42fecd1c1c98c4daecd249267122cbe4aed27cc04e7faa84d3d3e8`). Its three-scenario test evaluation is running on GPU0; the evaluation uses the unchanged test profile and capped detector allocation.

Seed430 checkpoint 00040 test evaluation also failed all three scenarios: 0/3 successes, 3/3 collisions, mean reward -32.36, SPL 0, paper ITR 0.01212, paper SD 0.4190 m; gate intervention fraction 18.1%. Metrics: `../experiments/eval_safe_clearance_seed430_final40_test_all3/metrics.json`. Both seed429 and seed430 final checkpoints thus failed the full test split despite prior training reward/validation differences.

To localize whether PPO training caused this regression, three-scenario test evaluations of both seeds' checkpoint 00020 are now running in parallel: seed429 on GPU0 and seed430 on GPU1. These are diagnostic checks, not additional training.

## Intermediate checkpoint diagnosis (2026-11-26 19:22 UTC)

Seed429 checkpoint 00020 also failed its three test scenarios (0/3 success, 3/3 collisions, mean reward -27.51, SPL 0, paper ITR 0.02246, SD 0.5212 m). Seed430 checkpoint 00020 passed all three (3/3, no collisions, mean reward 11.13, SPL 1.0, paper ITR 0, SD undefined). Thus seed430's later 00040 checkpoint regressed from its 00020 test result; the latter remains only a small-sample candidate. Metrics: `../experiments/eval_safe_clearance_seed429_update20_test_all3/metrics.json` and `../experiments/eval_safe_clearance_seed430_update20_test_all3/metrics.json`.

Additional test checks are running for seed429 checkpoint 00010 and seed430 checkpoint 00030 to bracket seed-specific changes during training. They use the same three test scenarios.

Seed430 checkpoint 00030 also passed the three test scenarios (3/3, no collisions, mean reward 9.87, SPL 1.0, paper ITR 0, SD undefined; gate intervention 11.3%). Its 00020 and 00030 checkpoints pass, while 00040 fails; this points to late-update degradation rather than a consistent seed430 inability. Metrics: `../experiments/eval_safe_clearance_seed430_checkpoint_test_all3/metrics.json`. Seed429 checkpoint 00010 test evaluation is still running.

Seed429 checkpoint 00010 also failed the three test scenarios (0/3 success, 3/3 collisions, mean reward -30.61, SPL 0, paper ITR 0.04177, SD 0.4739 m; gate intervention 20.9%). Seed429 has now failed test evaluation at checkpoints 00010, 00020, and 00040, while seed430 passed 00020 and 00030 but failed 00040. A seed430 checkpoint 00020 validation-split evaluation is running to add an independent split check for its best observed checkpoint.

Seed430 checkpoint 00020 validation evaluation was 2/3 successes and 1/3 collisions (mean reward -2.86, SPL 0.667, paper ITR 0.01062, SD 0.4539 m; gate intervention 23.0%). Combined with its test pass, it scored 5/6 across the validation and test splits, but both sets contain only three illustrative scenarios; this is insufficient to qualify a robust policy. A matching validation evaluation of checkpoint 00030 is running for comparison. Repeated checkpoint use of the test split is diagnostic and must not be treated as an untouched final selection set.

Seed430 checkpoint 00030 passed validation too: 3/3 successes, zero collisions, mean reward 11.12, SPL 1.0, paper ITR 0, SD undefined, gate intervention 12.7%, and fifth-percentile minimum human clearance 0.245 m. Together with its test pass, it achieved 6/6 over the three validation and three test layouts. Treat this only as a promising candidate: both splits contain just three illustrative layouts, and repeated post-hoc test checkpoint diagnostics mean the test split is no longer pristine for model selection. New scenario layouts and a frozen one-time evaluation are needed before stronger claims. Metrics: `../experiments/eval_safe_clearance_seed430_update30_validation_all3/metrics.json`.

## Fresh six-layout generalization evaluation started (2026-11-26 19:35 UTC)

Created `../mission/profiles/waffle_gazebo_illustrative_safe_clearance_fresh_test_v2.json`, containing six new four-human test layouts with varied robot lateral start/heading, human lane offset, speed, and travel direction. Train/validation retain the v1 safe-clearance scenarios. Profile SHA256: `7bce2f5999e39e56b3e1bf2f9438bcb61c77db937579ee5cc334d637ca4d7a2d`. The first draft had pedestrian start overlaps, caught by scene validation before any episode; it was corrected and JSON-level pair-distance checks now pass.

The repository's native evaluator strictly binds checkpoints to the complete profile SHA. To preserve that contract while testing new scenes, a mission-only adapter verifies that the v1 and fresh-test profiles match in every field except `scene` and `mission_note`, verifies the checkpoint's original v1 profile SHA and all other checkpoint/action/model contracts, then evaluates with the fresh scene profile. It does not edit the checkpoint or repository source. A six-episode single evaluation of seed430 checkpoint 00030 is running on GPU0. The prior attempt stopped at the expected checkpoint profile-mismatch guard before any episodes; it produced no results.

## Fresh-layout generalization result and broader training started (2026-11-26 19:49 UTC)

Seed430 checkpoint 00030 achieved only 3/6 successes on six new parametric test layouts; 3/6 ended in collisions. Mean reward -13.08, SPL 0.50, paper ITR 0.01488, paper SD 0.5109 m, gate intervention fraction 30.0%, and fifth-percentile minimum human clearance -0.029 m. Metrics: `../experiments/eval_safe_clearance_seed430_update30_fresh_test_v2_retry/metrics.json`. The promising 3/3 scores on each of the earlier three-layout validation/test splits did not generalize to this larger, varied profile.

Created diverse v3 profile `../mission/profiles/waffle_gazebo_illustrative_safe_clearance_diverse_v3.json` (SHA256 `f2c2ab8a0a5b70a8b4a7c172fff26983603a8542a696f77898c39086591fdb13`): 9 train, 9 validation and 6 separate test scenarios. Robot lateral start/heading and human lane offsets, speeds and directions vary; starts are checked for overlap. All remains explicitly illustrative and uncalibrated. Seeds431/GPU0 and 432/GPU1 are now training with the existing per-process 10% VRAM cap, 8 Gazebo envs each. Both have reached at least 480 transitions at this snapshot; the v3 test split remains reserved until validation-based checkpoint selection.

Operational note: initial launch attempts encountered a command sequencing/interrupt issue. Partial outputs were preserved, exact orphan Gazebo containers from the mis-seeded attempt were stopped by their unique log-volume path, and the current correctly seeded retry run_args were verified as 431 and 432. No unrelated process was targeted.

Diverse-profile progress (2026-11-26 19:50:52 UTC): seed431/GPU0 reached 720/9,840 transitions (one recent training episode, reward mean/median 11.5/11.5); seed432/GPU1 reached 1,440/9,840 (recent reward mean/median -14.6/-8.0). Both processes are active; free VRAM is 53,799 MiB on GPU0 and 35,635 MiB on GPU1. These are early noisy rewards, not policy evaluations.

Diverse-profile progress (2026-11-26 19:51:54 UTC): seed431/GPU0 reached 1,440/9,840 transitions (recent mean/median reward -22.0/-26.6); seed432/GPU1 reached 1,920/9,840 (-11.7/6.0). Mixed early returns continue; both remain active with ample free VRAM (53,799 MiB on GPU0 and 35,629 MiB on GPU1).

Diverse-profile progress (2026-11-26 19:53:02 UTC): seed431/GPU0 reached 2,160/9,840 transitions (recent mean/median reward -17.0/-26.4); seed432/GPU1 reached 2,640/9,840 (-6.7/7.0). Both jobs remain alive and advancing. Free VRAM: 53,789 MiB on GPU0, 35,577 MiB on GPU1.

Diverse-profile progress (2026-11-26 19:54:06 UTC): seed431/GPU0 reached 2,880/9,840 transitions (recent mean/median reward -17.9/-25.7); seed432/GPU1 reached 3,360/9,840 (-9.2/7.0). Both remain active; free VRAM was 53,599 MiB and 35,565 MiB.

Diverse-profile progress (2026-11-26 19:55:07 UTC): seed431/GPU0 reached 3,600/9,840 transitions (recent mean/median reward -16.0/-25.2); seed432/GPU1 reached 4,080/9,840 (-8.5/7.0). Both processes remain active; free VRAM was 53,579 MiB and 35,545 MiB.

Diverse-profile progress (2026-11-26 19:56:09 UTC): seed431/GPU0 reached 4,320/9,840 transitions (recent mean/median reward -13.1/-22.3); seed432/GPU1 reached 4,800/9,840 (-11.5/6.0). Both remain active; free VRAM was 53,579 MiB on GPU0 and 35,509 MiB on GPU1.

Diverse-profile progress (2026-11-26 19:57:14 UTC): seed431/GPU0 reached 4,800/9,840 transitions (recent mean/median reward -14.4/-23.8); seed432/GPU1 reached 5,520/9,840 (-15.2/-24.8). Both processes remain active. The negative episode averages and occasional low returns are being tracked as training evidence, not treated as a conclusion before held-out evaluation.

Diverse-profile progress (2026-11-26 19:58:14 UTC): seed431/GPU0 reached 5,520/9,840 transitions (recent mean/median reward -18.6/-25.5); seed432/GPU1 reached 6,240/9,840 (-17.8/-25.1). Both remain active; free VRAM was 53,575 MiB and 35,485 MiB. Reward averages are currently negative on the expanded train distribution, so the broader validation results will be important.

Diverse-profile progress (2026-11-26 19:59:37 UTC): seed431/GPU0 reached 6,240/9,840 transitions (recent mean/median reward -14.8/-22.2); seed432/GPU1 reached 7,200/9,840 (-17.7/-25.1). Both remain active. Negative training returns persist on the expanded distribution.

Diverse-profile progress (2026-11-26 20:00:37 UTC): seed431/GPU0 reached 6,960/9,840 transitions (recent mean/median reward -16.2/-22.4); seed432/GPU1 reached 7,920/9,840 (-20.2/-26.9). Both active; free VRAM was 53,567 MiB and 35,485 MiB.

Diverse-profile progress (2026-11-26 20:01:39 UTC): seed431/GPU0 reached 7,440/9,840 transitions (recent mean/median reward -16.2/-23.8); seed432/GPU1 reached 8,400/9,840 (-21.0/-29.5). Both remain active; recent training rewards are poor, which is a negative signal but still not a substitute for held-out validation.

Diverse-profile progress (2026-11-26 20:02:52 UTC): seed431/GPU0 reached 8,400/9,840 transitions (recent mean/median reward -18.1/-25.5); seed432/GPU1 reached 9,360/9,840 (-20.3/-27.6). Seed432 is near completion; both processes still active.

Diverse-profile seed432 training completed: 9,840 transitions, checkpoint 00040 SHA256 `e10d29fb6cd95e11c22479ed61dcb9d9c063524f89b81556ecf6d7d43578a200`. Its recent training mean/median rewards were -20.5/-28.7, a poor training signal. Nine-scenario v3 validation evaluation is running on GPU1. Seed431/GPU0 reached 8,880/9,840 transitions (recent mean/median -17.8/-25.5) and remains active.

Diverse-profile seed431 training completed at 9,840 transitions. Checkpoint 00040 SHA256: `713144f48bab10236510ec82e4a76a13a305e9cdcec744ab407f31b8dad6cb09`; final recent training mean/median reward -18.3/-25.9. Both seed431 and seed432 checkpoint 00040 validation evaluations are now running on their respective GPUs, nine validation scenarios per seed. Each final policy remains unqualified pending these validation results and the reserved six-layout test.

Validation polling (2026-11-26 20:06:39 UTC): seed431 final checkpoint has 2/9 validation episode logs; seed432 has 5/9. Both evaluators remain active. Free VRAM: 57,727 MiB on GPU0 and 36,627 MiB on GPU1.

Seed432 final checkpoint validation failed on the broader v3 split: 3/9 successes and 6/9 collisions, mean reward -17.97, SPL 0.333, paper ITR 0.04233, SD 0.4272 m, gate intervention fraction 22.8%, and fifth-percentile minimum human clearance -0.089 m. Metrics: `../experiments/eval_safe_clearance_seed432_final40_validation_all9/metrics.json`. Do not spend the reserved v3 test split on this checkpoint. Seed431 validation has 6/9 episode logs so far and is still running.

Seed431 checkpoint 00040 validation polling (2026-11-26 20:12:08 UTC): 7/9 scenario episode logs are complete; aggregate metrics are pending.

Seed431 final checkpoint also failed the v3 validation split: 2/9 successes, 6/9 collisions, 1/9 timeout, mean reward -25.80, SPL 0.222, paper ITR 0.03928, SD 0.4942 m, gate intervention fraction 31.2%, and fifth-percentile minimum human clearance -0.052 m. Metrics: `../experiments/eval_safe_clearance_seed431_final40_validation_all9/metrics.json`. Neither final diverse-profile run qualifies; the v3 test split remains unconsumed.

Seed431 final checkpoint validation failed: 2/9 successes, 6 collisions, and 1 timeout; mean reward -25.80, SPL 0.222, paper ITR 0.03928, SD 0.4942 m, gate intervention 31.2%, and p05 human clearance -0.052 m. Full metrics: `../experiments/eval_safe_clearance_seed431_final40_validation_all9/metrics.json`. Both seed431/432 final v3 checkpoints therefore fail validation. I am screening their 00020 checkpoints on the same three fixed-seed validation episodes, then will screen 00030 if needed; the v3 test remains untouched.

Checkpoint 00020 fixed-seed validation screen (three same-seed validation episodes) failed for both: seed431 0/3 successes and 3/3 collisions (mean reward -29.60, SPL 0, ITR 0.0826, SD 0.4885 m); seed432 1/3 success and 2/3 collisions (mean reward -16.91, SPL 0.333, ITR 0.0724, SD 0.5026 m). These are screen results on a subset, not full nine-layout validation. Matching checkpoint 00030 screens are now running; v3 test remains untouched.

Checkpoint 00030 validation screen (three fixed-seed scenarios) scored 2/3 successes for both seeds, with one collision each. Seed431 mean reward -1.75, SPL 0.667, ITR 0.04762, SD 0.4930 m; seed432 mean reward -1.34, SPL 0.667, ITR 0.04762, SD 0.4882 m. Full nine-layout validation for checkpoint 00030 is now running for both seeds; this is the validation-selected candidate check before any use of the reserved test.

Full checkpoint 00030 validation polling (2026-11-26 20:21:48 UTC): both seed431 and seed432 have four episode logs out of nine; aggregate metrics are pending. Free VRAM is 59,720 MiB on GPU0 and 36,575 MiB on GPU1.

Both checkpoint 00030 full v3 validation runs scored 6/9 successes, 3/9 collisions, no timeouts, and SPL 0.667. Seed431: mean reward -6.78, paper ITR 0.01962, SD 0.4926 m, gate intervention 22.5%, p05 minimum human clearance -0.023 m. Seed432: mean reward -6.13, ITR 0.02396, SD 0.5002 m, gate intervention 22.7%, p05 clearance -0.019 m. Full metrics: `../experiments/eval_safe_clearance_seed431_cp30_validation_all9/metrics.json` and `../experiments/eval_safe_clearance_seed432_cp30_validation_all9/metrics.json`. Both still have three collisions; the v3 test remains unused.

## Reward-shaping v4 experiment launched (2026-11-26 20:29 UTC)

The v3 checkpoint 00030 validation candidates still had three collisions each, so no v3 test evaluation was run. To target close human approaches and reliance on gate overrides, created reward-only v4 profile `../mission/profiles/waffle_gazebo_illustrative_safe_clearance_diverse_v4_reward.json` (SHA256 `e07056f6f8c5470627828ffd7387bea134ed1691989673635a028522af70e81c`). It keeps v3 scene geometry/splits identical and raises intervention cost 0.5→2.0, human-clearance cost 2.0→5.0, and clearance margin 0.25→0.4 m; all values remain illustrative, not calibrated.

Paired seeds431/GPU0 and 432/GPU1 are training from scratch with this reward profile, 10k requested steps, eight Gazebo envs each, 10% process VRAM cap. Seed432 reached 240 transitions at launch verification; seed431 was initializing. The six test layouts are still reserved.

Reward-v4 progress (2026-11-26 20:31:09 UTC): both seeds reached 1,200/9,840 transitions. Seed431 recent reward mean/median -36.4/-42.8; seed432 -27.0/-22.7. These early returns are worse than hoped under the stronger penalties; both runs remain active for more evidence. Free VRAM was 58,670 MiB (GPU0) and 35,523 MiB (GPU1).

Reward-v4 progress (2026-11-26 20:32:14 UTC): seed431 and seed432 both reached 1,920/9,840 transitions. Recent reward mean/median: seed431 -31.3/-41.0; seed432 -36.8/-43.0. Early rewards remain substantially negative; no checkpoint has been validated.

Reward-v4 checkpoint 00010 validation screens (2026-11-26T20:38:06Z): seed431 scored 0/3 successes with 3 collisions (mean reward -38.30, paper ITR 0.08069, SD 0.4867 m); seed432 scored 2/3 successes and 1 collision (mean reward -9.21, ITR 0.05747, SD 0.4986 m). Both relied on gate intervention in all three episodes (episode-level fraction 1.0). This is a small screen only, not full validation; the v3 test remains sealed. Both v4 runs continued through 5,520/9,840 planned transitions with recent reward mean/median -41.8/-42.0 (seed431) and -38.5/-42.1 (seed432). Checkpoint 00020 three-episode validation screens are running.

Checkpoint-20 reward-v4 validation screens (corrected, 2026-11-26T20:44:03Z): the first launch reused occupied v3 metrics paths and the evaluator rejected it; those old v3 files were preserved. Corrected screens wrote to unique v4 paths. Seed431: 0/3 successes, 3 collisions, mean reward -44.71, ITR 0.04390, SD 0.4847 m. Seed432: 2/3 successes, 1 collision, mean reward -8.56, ITR 0.05747, SD 0.4918 m. Both show gate interventions in all three episodes. Checkpoint-30 screens are running. Training seed432 completed 9,840 transitions (recent reward mean/median -38.7/-36.5); seed431 was at 9,120/9,840 (mean/median -43.9/-40.5). The v3 test remains sealed.

Reward-v4 training completion and checkpoint-30 screens (2026-11-26T20:45:37Z): both seed431 and seed432 completed 9,840 transitions; last reported recent reward mean/median was -43.5/-40.0 and -38.7/-36.5, respectively. Three-episode validation screens: seed431 0/3 successes, 3 collisions (mean reward -39.26, ITR 0.03762); seed432 1/3 success, 2 collisions (mean reward -33.08, ITR 0.06498). Gate intervention episode fraction was 1.0 in both. Final checkpoint 00040 full nine-layout validation is running for both seeds; the six-layout test split remains untouched.

New paired v3 replications launched (2026-11-26T20:49:33Z): seeds433/GPU0 and434/GPU1 are training from scratch on the same diverse v3 profile (9 train, 9 validation, 6 reserved test layouts), 9,840 planned transitions each, eight environments, with 10% per-process allocator caps. run_args.json confirmed both seeds and the v3 profile. Initial free VRAM after launch was 60,796 MiB GPU0 and34,751 MiB GPU1; no existing processes were changed. Final40 v4 full validation for seeds431/432 continues.

Active-run poll (2026-11-26T20:50:59Z): v4 final checkpoint 00040 nine-layout validation still running for both seeds. New v3 replication progress: seed433 960/9,840 transitions (recent reward mean/median -14.7/-27.8); seed434 1,200/9,840 (-10.5/-6.2). Free VRAM 60,582 MiB GPU0 and 34,501 MiB GPU1. No OOM; unrelated processes untouched; reserved test remains unused.

Reward-v4 final checkpoint 00040 full validation, seed431 (2026-11-26T20:57:01Z): 0/9 successes, 5 collisions, 4 timeouts; mean reward -83.00. Paper ITR 0.03097, SD 0.4788 m, gate intervention episode fraction 1.000. This candidate fails validation decisively; seed432 full validation is still running. The six-layout test split remains unused.

Reward-v4 final checkpoint 00040 full validation complete (2026-11-26T21:00:03Z), nine unique validation layouts per seed. Seed431: 0/9 successes, 5 collisions, 4 timeouts, mean reward -83.00, SPL 0, paper ITR 0.03097, SD 0.4788 m, gate intervention appeared in 9/9 episodes, p05 human clearance -0.0465 m. Seed432: 0/9 successes, 7 collisions, 2 timeouts, mean reward -56.29, SPL 0, ITR 0.03479, SD 0.4884 m, gate intervention appeared in 9/9 episodes, p05 clearance -0.0260 m. V4 does not qualify; its six held-out v3 test layouts were not evaluated. The new diverse-v3 replications (seeds433/434) continue, with their own validation and held-out v3 test set kept distinct.

V4 full validation has now failed for both seeds (recorded separately above). Checkpoint 00030 screens have been launched for the new v3-profile replications: seed433/GPU0 and434/GPU1, three fixed-seed validation episodes each. Both checkpoints exist, run_args confirm the correct independent seeds/profile, and initial free VRAM after launch was 60,560 MiB GPU0 and34,849 MiB GPU1. These are subset screens only; full validation and the reserved test remain pending.

New v3 replications checkpoint 00030 validation screens (2026-11-26T21:04:06Z): both scored 2/3 successes, 1 collision. Seed433 mean reward -1.32, paper ITR 0.04762, SD 0.4888 m; seed434 -2.76, ITR 0.04762, SD 0.4901 m. Gate intervention appeared in all three episodes. These are fixed-seed subset screens; paired nine-layout full validation is now running. Training seed433 completed 9,840 transitions; seed434 reached 9,600/9,840. Held-out test remains sealed.

New v3 replication training completed (2026-11-26T21:04:57Z): seeds433/434 each completed 9,840 transitions. Recent training reward mean/median: seed433 -13.4/-21.4; seed434 -16.7/-24.6. Checkpoint 00040 validation screens (three fixed-seed episodes per seed) are now running in parallel with full nine-layout checkpoint 00030 validation. Free VRAM after launch: 61,716 MiB GPU0 and35,897 MiB GPU1. All test layouts remain reserved.

V3 replication checkpoint 00040 screens (2026-11-26T21:07:45Z): both seeds433 and434 scored 2/3 successes, 1 collision on the same fixed-seed validation subset; mean rewards -2.52 and -2.59; ITR 0.04762 each; intervention appeared in 3/3 episodes. This suggests both cp30 and cp40 merit full validation despite screen limitations. Paired nine-layout cp40 validation is now running alongside cp30 full validation. VRAM remains ample (61,716 MiB GPU0, 35,897 MiB GPU1); reserved tests untouched.

V3 replication seed434 checkpoint 00030 full validation complete (2026-11-26T21:10:24Z): 7/9 successes, 2 collisions, no timeouts; mean reward 1.43, SPL 0.778, paper ITR 0.02850, SD 0.5045 m, p05 min human clearance -0.0134 m. Gate intervention appeared in all 9 episodes. Seed433 checkpoint30 and both checkpoint40 full validations remain running. This is a promising validation result but includes two collisions and safety-gate use throughout; it is not yet a held-out test candidate. Six test layouts remain sealed.

V3 replication seed433 checkpoint 00030 full validation complete (2026-11-26T21:11:38Z): 6/9 successes, 3 collisions, no timeouts; mean reward -11.51, SPL 0.667, paper ITR 0.02397, SD 0.4796 m, p05 human clearance -0.0466 m. Gate intervention appeared in all 9 episodes. Seed434 cp30 scored 7/9 successes and 2 collisions. Final checkpoint 00040 full validations for both seeds continue. No test layouts consumed.

V3 replication checkpoint 00040 full validation complete (2026-11-26T21:16:23Z): seed433 scored 6/9 successes and 3 collisions (mean reward -10.55, SPL 0.667, ITR 0.01893, SD 0.5065 m, p05 clearance -0.0226 m); seed434 scored 8/9 and 1 collision, no timeouts (mean reward 4.12, SPL 0.889, ITR 0.01587, SD 0.4883 m, p05 clearance -0.0014 m). Gate intervention appeared in every episode for both seeds. Selected seed434 checkpoint00040 using validation only for one final six-layout v3 test pass (PID3916404) because it was the best validation checkpoint across the new replications/checkpoints. This test is exploratory; no tuning/rerun after its result. Profile remains illustrative and uncalibrated.

Final one-time held-out test complete (2026-11-26T21:22:56Z): selected seed434 checkpoint00040 on 6 unique v3 test layouts. Outcomes: 4 ReachGoal, 1 Collision, 1 Timeout (success 0.667, collision 0.167, timeout 0.167), mean reward -3.83, SPL 0.667, paper ITR 0.00704, SD 0.5248 m, gate intervention appeared in all six episodes, p05 minimum human clearance +0.0247 m. The two failures mean this does not qualify as a robust policy. This profile remains illustrative and uncalibrated. The test split is consumed once; do not tune against it or rerun it. No mission training/evaluation process remains active. End free VRAM: 63,601 MiB GPU0, 40,520 MiB GPU1; unrelated user processes were left untouched.


## Paper-metric audit correction (2026-11-27 UTC)

The paper text (reference `paper.txt`, lines 305-319) defines ITR as an episode ratio c/C followed by the average ratio across testing episodes. SD is the average distance to the closest human over intrusion observations, with intrusion events determined by ground-truth human positions from t+1 through t+5. The current implementation now keeps per-episode ITR and averages episodes equally; SD now uses current-time closest-human center distance at each intrusion timestep and pools those distances by intrusion event. The corrected implementation has six focused unit tests, including unequal episode lengths/event counts.

This audit found that the previous held-out run's reported SD used future-window distances instead of current-time distances. Its stored per-timestep traces are unavailable, so the old SD value cannot be recomputed. The six-layout test remains consumed and must not be rerun; treat its reported SD as invalid/unavailable. The per-episode ITRs were already averaged equally and remain consistent with the paper definition, subject to the documented illustrative simulator semantics. The held-out outcome (4/6 success, one collision, one timeout) is unchanged and still fails qualification. New validation and evaluation must use the corrected metrics.


Corrected real-Gazebo metric integration check (2026-11-27 UTC): reran the same three validation scenarios after fixing both per-event SD and paper-faithful ITR aggregation. It completed 3/3 unique validation scenarios (2 reached goal, 1 collision) in 67.7s. Corrected mean ITR is 0.04761905 (unweighted mean of per-episode ratios); pooled SD is 0.880662 m over 4 intrusion timesteps. Evidence: `../experiments/eval_v3rep_seed434_cp40_itrfix_validation_screen3/metrics.json` and `../logs/eval_v3rep_seed434_cp40_itrfix_validation_screen3.stdout`. This is metric-path integration evidence only, not a broad performance validation. No held-out test data were used.


## Source commit and new train-only trial (2026-11-27 UTC)

- Metric, integration and handoff changes are committed locally as `2fce424` on `codex/waffle-gazebo-mission-20261126`; the working tree is clean. The branch push was rejected: the H100 GitHub SSH identity is `krishmarwah` and has no write permission to `Anurag9000/RL_Demo`; HTTPS has no credential helper. The branch remains local.
- Current VRAM snapshot at trial admission: GPU 0 had only about 1.2 GiB free, so it was left unused. GPU 1 had about 40.7 GiB free. One independent seed-435 PPO run was launched on GPU 1, train split only, 25,000 requested transitions, eight Gazebo worlds, 20% allocator cap; checkpoint/full-state output is isolated under the experiment root. It uses the illustrative v3 profile.
- Script name and `run_args.json` record the 25k budget. Its inherited output directory label contains `10k`; the machine-readable run arguments are authoritative. No validation/test split is used for training; no consumed test data were touched.


Seed 435 checkpoint 00010 validation screen (2026-11-27 UTC): 0/3 successes, 3 collisions, no timeouts across three unique validation scenarios. Corrected mean per-episode ITR 0.07161954; event-pooled SD 0.775017 m over 11 intrusion timesteps. This is a negative early-checkpoint screen, not a final evaluation. The train-only seed-435 run continues to its pre-set 25,000 transitions to retain a complete, auditable result. Metrics: `experiments/eval_seed435_cp10_validation_screen3/metrics.json`. At this observation, training had reached 5520/25,000 transitions; recent rolling episode reward was -26.46. No test layout was used.


Seed 435 checkpoint 00020 matched validation screen (2026-11-27 UTC): 0/3 successes, 3 collisions on the same three validation scenarios as checkpoint 00010. Corrected mean ITR 0.05961336; pooled SD 0.797537 m across 13 intrusion timesteps. This does not show early improvement. Training remains on its train-only 25k budget; final selection still must use a larger validation bank, with test split sealed. Metrics: `experiments/eval_seed435_cp20_validation_screen3/metrics.json`.


## 2026-11-27T01:12:52Z continuation snapshot

- Source checkout is clean at local commit 2fce424f1fbac6b6dbd205e6e7c273e1fb1186b4 on codex/waffle-gazebo-mission-20261126; push remains blocked by the authenticated GitHub identity lacking repository write access.
- Seed 435 training is still running on GPU 1, PID 3960750; latest log at 01:12:31Z reports 13,680/25,000 transitions (12 FPS), rolling mean/median -29.1/-30.8. GPU 1 had 44,671 MiB free at 01:11:19Z; GPU 0 had 1,189 MiB free and remains untouched due another user allocation.
- Validation checkpoint 10 and 20 each had 0/3 successes and 3 collisions. The profile has only 9 validation and 6 test scenarios, versus required 500-case independent banks. No qualification claim is supportable.
- Semantic review ledger now documents dedicated reads of calibration/identification, perception/tracking, PPO/storage/training state, Gazebo environment/worker/world, ROS2 inference/gate, and the pinned reference submodule. This remains partial coverage of a 262-entry parent repository (180 Python files, about 23,756 source lines); most files remain unreviewed.
- The main mission gates remain open: independent scenario banks, broad convergence/held-out evaluation, CPU/GPU/H100 scale benchmarks, measured hardware calibration, deployment qualification, and upstream branch publication/CI confirmation.


## H100 PPO throughput microbenchmark (2026-11-27 UTC)

- Completed two isolated 4-world Gazebo PPO benchmark runs on physical GPU 1, each 30-step rollout, one warmup and two measured updates (240 transitions), while seed 435 training continued. Eager measured 10.0479 transitions/s (rollout 23.5921s; GAE/bootstrap 0.01109s; learner 0.28240s). Scan-GAE plus masked-GRU measured 10.0326 transitions/s (rollout 23.6235s; GAE/bootstrap 0.01027s; learner 0.28824s). No observed speedup; sample is too short and concurrent to claim a robust ranking. Gazebo rollout/physics/detector dominates; learner is not the bottleneck in this sample. PyTorch peak allocated was about 0.4 GiB. Detailed JSON is in experiments/benchmark_h100_gpu1_eager2_20261127/benchmark.json and experiments/benchmark_h100_gpu1_scan_20261127/benchmark.json.
- Setup logs preserve and explain two quickly corrected launch attempts: unsupported --num-threads option and an invalid Cartesian-init request against a native checkpoint. No job ran for either failed attempt.

## Latest status at 2026-11-27T01:19:27Z

- Seed 435 remains alive (PID 3960750), 19,200/25,000 transitions, 12 FPS; rolling training mean/median reward -28.0/-31.7. GPU 1 had 44,631 MiB free; GPU 0 had 1,155 MiB free. Unrelated workloads remain untouched.
- Audit coverage remains partial and qualification blockers remain as listed above. No 500-case bank, measured calibration, full scale benchmark, held-out qualification, or deployable/actuation claim is complete.


## Seed 435 training completion and v5 bank preparation (2026-11-27T01:32:30Z)

- Seed 435 completed all 104 PPO updates / 24,960 transitions, as proven by training_summary.json. Final checkpoint is 00103.pt. Last rolling 100 episode mean/median reward was -24.8/-31.3. This is a poor training result, not convergence evidence.
- Final checkpoint validation on all 9 v3 validation cases is running as PID 4005945, with eight Gazebo workers. No test data were used.
- A reproducible v5 profile now has 500 train, 500 validation and 500 test cases for the four-human illustrative condition; profile geometry and split IDs passed. Hash and sampler are documented in CODE_AUDIT_PROGRESS.md. This does not establish measured fidelity or broad behavioral validity.
- Prepared seed 436 fresh PPO training on the 500-case v5 train bank for 250,000 transitions, 8 Gazebo worlds on physical GPU 1 at 20% PyTorch allocator cap, and a frozen 500-case test evaluation script. The training has not started pending final v3 validation completion and fresh VRAM admission.
- Outstanding: finish v3 validation, launch/monitor seed 436, select using new validation bank, preserve test split until checkpoint lock, then run 500-case test once. Other campaign deliverables remain open.


## Seed 435 final v3 validation (2026-11-27 UTC)

- Checkpoint 00103 evaluated on all 9 unique v3 validation scenarios: 6 reach-goal, 2 collision, 1 timeout (success 66.7%, collision 22.2%). Mean paper ITR 0.02491; event-pooled SD 0.89669m over 12 intrusion observations; p05 minimum human clearance -0.0262m; gate intervention in all episodes. Elapsed 308.0s across eight Gazebo workers. Evidence: experiments/eval_seed435_final_v3_validation9/metrics.json and logs/eval_seed435_final_v3_validation9.stdout. No test split was used. Policy remains unqualified.

## Seed 436 v5 500-bank PPO started

- Started fresh native-action PPO training on the v5 500-case train split, 250,000 requested transitions, 8 Gazebo worlds, seed 436, 20% PyTorch allocator cap, physical GPU 1, process PID 4014103. At admission GPU 1 reported 46,222 MiB free; GPU 0 remained untouched. Output and worker logs are isolated as specified in the mission script. This will be a long-running training campaign; monitor until completed before selecting validation checkpoint.

## 2026-11-27T01:47Z continuation update

- Seed-436 v5 training resumed from the 8-world update-0 state into a 16-world rollout, using the same train-only 500-scenario bank, physical GPU 1, and 20% per-process allocator cap. Live PID 4076346; at this snapshot it had completed updates 0 and 1 in the b16 segment (960 segment transitions; continuation starts from the saved 240-transition seed state). Reported FPS is 3. This is far below the 250,000-transition target's practical runtime, so no completion-time claim is made. The run remains live and its checkpoints/state are isolated under `experiments/waffle_safe_clearance_gpu1_ppo_seed436_v5_250k_b16_20261127/`.
- GPU snapshot at 01:46:15Z: H100 GPU 1 had 38,867 MiB free; GPU 0 remained occupied by unrelated work and was not touched. GPU utilization is recorded only as a diagnostic; admission was based on free VRAM.
- The v5 validation/test splits remain unused by this training continuation. No candidate selection or final held-out evaluation has occurred. Do not treat this early training snapshot as qualification evidence.

At 01:49:34Z the seed-436 b16 process remained alive at update 2 (1,440 segment transitions), reporting 3 FPS. Training is active but slow; no change to the train-only configuration, VRAM cap, or GPU assignment was made.

## Objective and audit continuation note

The complete attached master mission objective was re-read on 2026-11-27. It contains overlapping prompt sections; the stricter common requirements remain in force, including Gazebo-only target physics, no physical actuation, 600-second stable health-check cadence, exact trace-based paper metrics, and explicit nominal-vs-measured separation. The new source line references and untested legacy edge cases are recorded in `mission/CODE_AUDIT_PROGRESS.md`; they are not claimed as reproduced failures.

## Repository/CI refresh (2026-11-27T01:56:08Z)

- On H100, fetched `origin/main` without modifying the mission worktree. `origin/main` is still `e5a6da0a938a581d91e6ac7f266bb7c7bf105611`; mission branch HEAD `2fce424f1fbac6b6dbd205e6e7c273e1fb1186b4` is one commit ahead and has no uncommitted tracked changes. No newer upstream commit was found.
- Public GitHub Actions API reports run 45814045755 (Gazebo Harmonic native validation) and 45814045744 (Waffle tensor validation) completed successfully on e5a6da0a938a. These checks do not cover mission-branch changes or full detector-in-loop H100 training. `gh` is not installed on the server; read-only public API query succeeded. CI on 2fce424 remains unverified because that commit is not published.

## Consolidated health sample (2026-11-27T02:00:17Z)

- Seed-436 v5 process PID 4076346 remained live at elapsed 18m29s. Progress advanced from 1,440 to 3,840 segment transitions (8 updates logged, each 480 segment transitions); reported FPS remains integer-rounded 3. Latest reward/loss fields are finite. This is still an early pilot, not a convergence signal.
- The process has not yet reached its next configured save interval: the only learner checkpoint is `00000.pt`; `training_state.pt` timestamp is 01:44:13Z. Current unsaved interval should be preserved until an update-boundary checkpoint if the run remains healthy.
- H100 physical GPU 1 free VRAM: 38,813 MiB; physical GPU 0 free VRAM: 63,835 MiB. GPU 0 remains untouched. Host reports 407 GiB available memory, 469 MiB swap used, 6.8 TiB disk free, and 252 GiB `/dev/shm` free. Capacity remains ample for the current run.
- Next stable health check due no earlier than 02:10:17Z. Do not recheck during the idle interval; continue source/paper analysis and durable documentation meanwhile.


## Recovery rerun and startup fault instrumentation (2026-11-27T06:26:41Z)

- Both prior v5 learners had terminated at a Gazebo worker stdout EOF while warming the world on reset. Replayed exact failed scene/config pairs independently; each passed. Root cause remains unconfirmed.
- Modified server checkout at base commit `2fce424f1fbac6b6dbd205e6e7c273e1fb1186b4`: `gazebo_sim/worker.py` now reports worker PID, exit code, log path and bounded stdout tail on EOF; `gazebo_sim/vector_env.py` retries a reset warm-start once for this specific EOF only, preserving the same scenario and recording attempts. This warm start is not an RL transition. Patch saved at `mission/patches/worker_startup_recovery.patch`.
- Added `test_startup_retries_same_scenario_once_without_transition`; `python -m unittest tests.test_waffle_gazebo_integration -v` passed all 4 tests (0.059 s). `git diff --check` passed. The first two test invocations exposed only test assertion mistakes (log file nested deeper; vector env exposes `worlds` directly); corrected and final run passed.
- Resumed seed 436 from its latest full state (10,320 transitions; 16 envs × 30 steps, 8 minibatches, seed 436) on physical GPU 1, and seed 437 from its latest full state (14,640 transitions; 8 envs × 30 steps, 4 minibatches, seed 437) on physical GPU 0. New isolated output/log roots use `recovery1`; each is capped at 20% PyTorch allocator fraction. Requested remaining budgets are 240,000 and 235,680 segment transitions respectively, targeting about 250,320 cumulative transitions per seed. No validation or test scenarios were used.
- Launch observed at approximately 2026-11-27T06:23:15Z. At 06:25Z, both Python supervisors were alive (PIDs 2042361 and 2042362); both had loaded the learner and launched Gazebo workers, but neither had written its first PPO progress row yet. No retry JSONL had been created at this snapshot. GPU VRAM: GPU 0 8,360 MiB used / 86,952 MiB free; GPU 1 11,698 MiB used / 83,614 MiB free. These totals include other processes; none were altered. Next stable-job health poll no earlier than 06:35Z.
- Work remains incomplete and unqualified.


## Full H100 validation after startup-recovery patch (2026-11-27T06:28:23Z)

`bash scripts/validate_waffle.sh` completed successfully on the server. It ran compileall across the listed repository modules and unittest discovery: 68 tests passed, 1 optional Python-RVO2 test skipped (`Python-RVO2 not installed`), exit code 0. This is source/unit validation and does not establish Gazebo long-run stability or policy performance. `git diff --check` passed. The patch file is the source diff against base commit `2fce424f1fbac6b6dbd205e6e7c273e1fb1186b4`; because it is already applied in the worktree, `git apply --reverse --check` passed. An ordinary forward `git apply --check` against the modified worktree reported expected context failures; no code was changed by that check.


## Recovery training first progress sample (2026-11-27T06:29:00Z)

Both resumed jobs advanced past world initialization. Seed 436 logged recovery-segment update 1 / 960 transitions (reported 3 FPS), so 11,280 cumulative transitions including its saved 10,320. Seed 437 logged recovery-segment update 1 / 480 (2 FPS), 15,120 cumulative including its saved 14,640. Rolling episode reward means were -25.2 and -24.9 respectively, early/noisy and not evidence of policy quality. Last observed retry JSONL absent (no retry event observed yet). Full validation completed while both jobs remained alive. Treat 06:28Z as first stable progress sample; next training/resource health check no earlier than 06:38Z.


## Worker EOF diagnostic regression and final validation count (2026-11-27T06:33:26Z)

Added a direct fake-process test for `Worker._read`: it verifies an EOF event includes PID, exit code, worker log path and bounded stdout tail while preserving a preceding framed response. Reran `bash scripts/validate_waffle.sh`: compileall passed, 69 tests passed, 1 optional Python-RVO2 test skipped, exit code 0 (2.653 seconds). Final mission patch artifact was refreshed; `git diff --check` and `git apply --reverse --check` pass. No training process was restarted or interrupted.


## 2026-11-27T06:39Z consolidated recovery health sample

After 10+ minutes since the first stable progress sample, both learners remain alive (seed436 Python 2042361 on GPU1; seed437 Python 2042362 on GPU0). Seed436 recovery segment: update 6, 3,360 transitions, reported 3 FPS, rolling episode reward mean -25.71; cumulative log-visible count 13,680 including 10,320 saved input transitions. Most recent full-state checkpoint persists 10,800 cumulative transitions; newer segment work is not all checkpointed yet. Seed437 recovery segment: update 8, 2,160 transitions, 2 FPS, rolling episode reward mean -25.63; cumulative log-visible count 16,800 including 14,640 saved input. Latest full state persists 14,880 cumulative transitions. No retry JSONL or exit records observed; both training-state timestamps advance. GPU0: 8,542 MiB used / 86,770 MiB free; GPU1: 7,769 MiB used / 87,544 MiB free. Host available memory 459 GiB; disk free 6.8 TiB. No unrelated process was touched. Next stable health poll no earlier than 06:49Z.

Per the requested short time window, current completed work is patch/test/restart plus this first healthy interval. The full training/evaluation mission remains active and will take substantially longer than 10 minutes at the measured 2-3 FPS. It is not complete or qualified.


## Held-out evaluation admission and selection gate (2026-11-27T06:51:35Z)

Found the previous seed-436 test launcher hard-coded to an obsolete non-recovery checkpoint/output path. Replaced it with a wrapper requiring the selected checkpoint, fresh output directory and validation-lock JSON; added a matching seed-437 wrapper and a generic v5 bank evaluation runner. The runner verifies train-manifest provenance, seed/action/backend, canonical v5 profile and hash, exactly 1,500 unique bank IDs and exactly 500 cases in the requested split. Test evaluation additionally requires a locked checkpoint SHA, 500-case validation evidence, explicit validation-only selection and confirmation that test was unused.

Negative admission smoke used an existing seed-436 v5 checkpoint without a lock: it exited before evaluation with `held-out test requires a validation lock JSON`; neither output directory nor request record was created. `bash -n` passed for all three launchers. No validation or test cases were run by this smoke. Training remains in progress; next scheduled live health sample is no earlier than 06:51Z.


## 2026-11-27T06:52Z consolidated recovery health sample

Both learners remain live (seed436 PID 2042361, GPU1; seed437 PID 2042362, GPU0; elapsed about 29 minutes). Segment progress: seed436 update index 12 / 6,240 transitions, rounded 3 FPS, rolling episode reward mean -24.91; seed437 update index 16 / 4,080 transitions, rounded 2 FPS, mean -24.75. Cumulative log-visible transitions are 16,560 and 18,720, respectively. Latest durable training states remain 10,800 and 14,880 cumulative transitions, with only segment checkpoint `00000.pt` written so far; later progress is currently unsaved until update 20. No retry logs or exit records exist. GPU0 has 86,770 MiB free and GPU1 91,507 MiB free. No unrelated processes were changed. Next health sample no earlier than 07:02Z.

Evaluation scripts now require validation-lock artifacts before the test bank can be opened. This is a readiness gate only; no validation/test episodes have run for these recovery models. Mission remains incomplete.


## Evaluation gate preflight correction (2026-11-27T10:16:24Z)

The first test-gate smoke exited 127 because the new script called `python` before venv activation; it created no output/request and did not enter train/evaluation. Changed only the metadata preflight to `python3`, then reran the same no-lock test gate: it exited 1 with the intended `held-out test requires a validation lock JSON`, and again created no output/request. The training invocation still runs under the activated model venv. `bash -n` passes. Test gate additionally requires matching training seed, canonical profile digest and nonempty selection criteria.

## 2026-11-27T10:24Z recovery training and profile-matched throughput sample

Both v5 recovery learners remain alive after roughly 4 hours elapsed. Seed 436 (GPU1, 16 worlds x 30 steps) reached recovery update 112 / 54,240 segment transitions (reported 3 FPS), with latest rolling reward mean -24.78. Including the 10,320-transition saved starting state, log-visible cumulative transitions are 64,560. Durable learner state timestamp is 15:28 IST (09:58Z), at checkpoint 00100; later progress remains volatile until the next atomic save. Seed 437 (GPU0, 8 worlds x 30 steps) reached recovery update 141 / 34,080 segment transitions (reported 2 FPS), latest rolling reward mean -25.33. Including the 14,640-transition saved starting state, cumulative is 48,720. Durable learner state timestamp is 15:52 IST (10:22Z), checkpoint 00140. No process was restarted or modified. GPU0 free VRAM 91,076 MiB; GPU1 free VRAM 89,004 MiB. This is an authorized residual-VRAM snapshot only; no exclusivity is claimed.

The separate v5 four-world H100 benchmark completed with exit 0. It loaded seed437 recovery checkpoint 00000, used 1 warmup + 2 measured updates, and recorded 240 measured transitions in 132.56s rollout (1.805 transitions/s). Learner update time was 0.381s total; rollout dominates. Summed Gazebo worker counters were reset 212.73 world-seconds, physics IPC 65.39, detector 5.03 and step 73.24 across 360 steps/51 resets. Peak PyTorch allocation was 1.29 GB. This was a short throughput probe, not a policy-quality evaluation; it does not beat the concurrent 8- and 16-world rates observed in the live runs.

An isolated install of the optional `rvo2` wheel failed because the configured package index has no matching distribution. The shared environment was not modified. Existing test validation remains 69 passed / 1 skipped for missing optional Python-RVO2; no parity result is claimed. Native-reference parity remains open. Full training, validation, final selection, and reserved test evaluation remain incomplete. Next training-health sample no earlier than 10:34Z.

## 2026-11-27T10:30Z upstream recheck

On H100, fetched `origin/main` read-only. Remote remains `e5a6da0a938a581d91e6ac7f266bb7c7bf105611`; mission branch HEAD remains `2fce424f1fbac6b6dbd205e6e7c273e1fb1186b4`, one local mission commit ahead. Existing three-file startup-recovery patch is still the only tracked worktree modification. No upstream advance was found; no merge/reset/pull was performed, preserving the exact source used by both active training jobs.

## 2026-11-27T10:34Z consolidated recovery health and worker-fault sample

After more than 10 minutes, both PPO learners remain alive. Seed436: update 117, 56,640 recovery-segment transitions (3 rounded FPS), rolling reward mean -24.90; cumulative log-visible count including the 10,320 saved start is 66,960. Latest durable checkpoint remains 00100 and training_state.pt timestamp 09:58Z, so later work is not yet durable. Seed437: update 147, 35,520 recovery-segment transitions (2 rounded FPS), rolling mean -24.90; cumulative including 14,640 saved start is 50,160. Latest durable checkpoint 00140, state timestamp 10:22Z. GPU0 free VRAM 91,070 MiB; GPU1 free VRAM 89,004 MiB. No process or unrelated job was modified.

The startup-recovery instrumentation has now recorded four one-time resets on seed436 (world slots 4, 5, 13, 15), at scenario episodes 457, 395, 382 and 289. Three worker processes exited 139 (SIGSEGV); one exit code was unavailable. All retries were marked retryable and training continued. The four corresponding worker logs contain only SDF `gz_frame_id` warnings and no stack trace or stdout tail. Seed437 has no nonempty retry JSONL observed. This is repeated native-worker failure evidence, not an isolated one-off; root cause remains unknown. Keep the current runs and artifacts intact and inspect a core/backtrace only if one can be captured without disrupting learners. Next health poll no earlier than 10:44:16Z.

### SIGSEGV evidence availability check (2026-11-27T10:36Z)

Read-only artifact search found no core file or dump in either recovery worker log tree, `/var/crash` is empty, and `coredumpctl` has no matching record. Both live Python learner processes expose a soft `Max core file size` limit of 0 bytes; the interactive shell also has `ulimit -c` 0 and kernel core handling routes through apport. This explains why no local core is available, but does not identify the SIGSEGV source. Enabling worker core capture would require a future worker launch/restart; the active jobs were left untouched and no restart is planned solely to obtain a core.

## 2026-11-27T10:44Z consolidated recovery health sample

Both recovery PPO processes remain alive at elapsed 4h21m. Seed436 (GPU1, 16 worlds) reached recovery update 121 / 58,560 segment transitions (rounded 3 FPS), rolling reward mean -24.91, cumulative log-visible transitions 68,880. Its durable checkpoint advanced to 00120 and training_state.pt timestamp 10:40Z. Seed437 (GPU0, 8 worlds) reached update 153 / 36,960 segment transitions (rounded 2 FPS), rolling mean -25.19, cumulative 51,600. Its durable checkpoint remains 00140 and training state 10:22Z; subsequent work is still unsaved pending its next save boundary. No new worker retry was recorded: the four seed436 retries remain the total; no seed437 retry or exit record was found. Free VRAM: GPU0 91,070 MiB; GPU1 90,660 MiB. The jobs and unrelated processes remain untouched. Next health poll no earlier than 10:54:31Z.

## 2026-11-27T10:46Z v5 32-world scaling probe launched

Prepared `mission/scripts/benchmark_h100_v5_32env.sh` from the earlier four-world probe and passed `bash -n`. It runs a separate 32-world/8-minibatch v5 Gazebo benchmark on GPU0, with one warmup and two measured updates, loading seed437's latest durable checkpoint 00140 and retaining the 20% PyTorch allocator cap. Output/log/status are isolated under the mission run root. Job PID 1255819 loaded the checkpoint successfully at 10:46Z; measured result is pending. This extra benchmark was authorized by the H100 residual-VRAM instruction and does not alter either learner. Main learner health poll remains due no earlier than 10:54:31Z.

### 32-world probe progress (2026-11-27T10:50Z)

The isolated probe has 32 worker manifests and has completed warmup update 0: 960 transitions, reported 4 FPS, rolling reward mean -25.42. It has not yet produced the two measured benchmark updates or `benchmark.json`; no throughput conclusion is recorded yet. Probe PID 1255819 remains live. Main learner health polling remains on the 10-minute cadence.

## 2026-11-27T10:55Z consolidated recovery health and scale-probe sample

Both main learners remain alive after 4h32m. Seed436 (16 worlds, GPU1) reached update 125 / 60,480 segment transitions (rounded 3 FPS), rolling mean -25.04, log-visible cumulative 70,800. Durable checkpoint/state remain at update 120 / 10:40Z. Seed437 (8 worlds, GPU0) reached update 157 / 37,920 segment transitions (rounded 2 FPS), rolling mean -25.43, cumulative 52,560. Its durable checkpoint/state remain update 140 / 10:22Z. Retry count remains four on seed436 and zero on seed437; no exit records. GPU0 free VRAM 85,654 MiB (32-world probe has added about 5.4 GiB allocation); GPU1 free 90,660 MiB. Both training jobs and unrelated processes are preserved.

The separate 32-world/8-minibatch benchmark completed warmup plus measured update 1: 1,920 total transitions, measured update FPS 4, rolling reward mean -25.21. One measured update remains; benchmark JSON/exit record not yet written. This is throughput evidence only, not policy-quality evidence. No evaluation-request files exist, so no recovery validation or v5 held-out test episode was run. Next main-job health poll no earlier than 11:05:00Z.

## 2026-11-27T11:05Z consolidated recovery health sample

Both main recovery learners remain alive after about 4h42m. Seed436 (GPU1, 16 worlds) reached update 130 / 62,880 recovery-segment transitions (rounded 3 FPS; +2,400 segment transitions over the preceding 10 minutes), rolling reward mean -24.72. Latest durable checkpoint/state remain update 120 / 10:40Z; the next configured save is update 140. Seed437 (GPU0, 8 worlds) reached update 164 / 39,600 recovery-segment transitions (rounded 2 FPS; +1,680 over the preceding 10 minutes), rolling reward mean -25.02. Durable checkpoint/state advanced to update 160 / 10:59:43Z. No process restart or unrelated-job change occurred.

Free VRAM: GPU0 92,723 MiB of 95,830; GPU1 90,660 MiB of 95,830. Seed436 retains the four known startup retries; seed437 has no retry. The finished 32-world probe exited 0 but did record one worker startup retry in world slot 7, episode 6, scenario `v5-train-0328`; child exit code was 139. Thus the probe produced throughput metrics despite one recovered worker crash; 32-world stability is not established and the underlying SIGSEGV cause remains unknown. No evaluation request exists and no v5 validation/test episodes have run; held-out test remains sealed. Next main learner health sample no earlier than 11:15:18Z.

Counter semantics for the 11:05 sample: recovery-segment transitions exclude the saved continuation offsets (seed436 +10,320; seed437 +14,640). Cumulative log-visible totals are therefore 73,200 and 54,240, respectively; only checkpoint/state files cited above are durable.

## 2026-11-27T11:15Z consolidated recovery health sample

Both learners remain alive at elapsed 4h52m. Seed436 (GPU1, 16 worlds) reached update 134 / 64,800 recovery-segment transitions (rounded 3 FPS; +1,920 in 10 minutes), rolling reward mean -24.52. Durable state/checkpoint remain update 120 / 10:40Z; update 140 is next save boundary. Seed437 (GPU0, 8 worlds) reached update 169 / 40,800 recovery-segment transitions (rounded 2 FPS; +1,200 in 10 minutes), rolling mean -24.99. Durable state/checkpoint remain update 160 / 10:59:43Z. No new retry or process exit was observed; retry totals remain four seed436 / zero seed437.

Counter semantics: add saved continuation offsets 10,320 (436) and 14,640 (437), giving cumulative log-visible totals 75,120 and 55,440. Free VRAM is 91,594 MiB GPU0 and 90,660 MiB GPU1. No evaluation request exists; no v5 validation or held-out test episodes have run. Next main learner health sample no earlier than 11:25:32Z.

## 2026-11-27T11:25Z consolidated recovery health sample

Both learners remain alive at elapsed 5h02m. Seed436 (GPU1, 16 worlds) is at zero-based progress index j=139 / 500, segment transitions 67,200 (rounded 3 FPS; +2,400 over 10 minutes), rolling reward mean -25.09. Durable state/checkpoint remain 00120 at 10:40Z; j=140 is the upcoming save boundary. Seed437 (GPU0, 8 worlds) is at j=176 / 982, segment transitions 42,480 (rounded 2 FPS; +1,680 over 10 minutes), rolling mean -25.05. Durable state/checkpoint remain 00160 at 10:59:43Z. Both continue at the measured pace; completion of their configured budgets is many hours away at present rates.

Add continuation offsets 10,320 (436) and 14,640 (437): cumulative progress 77,520 and 57,120 of the shared 250,320-transition target. No new worker retry or exit was observed; totals remain four seed436 / zero seed437. Free VRAM: GPU0 92,723 MiB, GPU1 90,660 MiB. No evaluation request exists; v5 validation/test remain unrun and test sealed. Next main learner health sample no earlier than 11:35:49Z.


## 2026-11-27T11:44:41Z consolidated recovery health sample

Both main learners remain alive at elapsed 5h21m. Seed436 (GPU1, 16 worlds) reached update 149 / 72,000 recovery-segment transitions (rounded 3 FPS; +4,800 over 19 minutes), rolling reward mean -24.8. Seed437 (GPU0, 8 worlds) reached update 188 / 45,360 recovery-segment transitions (rounded 2 FPS; +2,880 over 19 minutes), rolling reward mean -25.6. Durable state timestamps: seed436 16:55:50 IST and seed437 17:01:46 IST; both remain older than current logs, so only those saved states are durable. Add continuation offsets 10,320 and 14,640: cumulative log-visible totals 82,320 and 60,000 of 250,320. No new retry file or evaluation request exists. Both PIDs 2042361/2042362 are alive. Free VRAM is 92,723 MiB on GPU0 and 90,660 MiB on GPU1. Held-out test remains sealed. Next health sample no earlier than 11:54:41Z.



## 2026-11-27T11:48:23Z code synchronization

Evaluation quota-stop fix and the pending Gazebo worker diagnostic updates are on commit `bf72547` (following campaign commit `2fce424`). Kakarot branch `codex/waffle-gazebo-mission-20261126` and GitHub branch `origin/codex/waffle-gazebo-mission-20261126` both point to `bf72547`; latest `main` remains unchanged. H100 fetched that pushed branch and is tracking it.


## 2026-11-27T11:54:52Z consolidated recovery health sample

Both main learners remain alive at elapsed 5h31m. Seed436 (GPU1, 16 worlds) reached update 153 / 73,920 recovery-segment transitions (rounded 3 FPS; +1,920 over 10 minutes), rolling reward mean -25.2. Seed437 (GPU0, 8 worlds) reached update 194 / 46,800 recovery-segment transitions (rounded 2 FPS; +1,440 over 10 minutes), rolling reward mean -25.5. Add saved continuation offsets 10,320 (436) and 14,640 (437): cumulative log-visible progress 84,240 and 61,440 of 250,320. Durable state remains at update boundaries 140 and 180 respectively; later updates are not checkpointed yet. No new retry record or evaluation request was found. Free VRAM remains 92,723 MiB GPU0 and 90,660 MiB GPU1. Held-out test remains sealed. Next health sample no earlier than 12:04:52Z.



## 2026-11-27T11:57:22Z targeted verification

Both evaluation quota tests passed under the campaign PYTHONPATH. No training process or GPU workload was changed.


## 2026-11-27T12:05:00Z consolidated recovery health sample

Both learners remain alive at elapsed 5h42m. Seed436 (GPU1, 16 worlds) reached update 158 / 76,320 recovery-segment transitions (rounded 3 FPS; +2,400 in 10 minutes), rolling reward mean -25.4. Its latest durable state is still update 140 (16:55:50 IST); j=160 is next save boundary. Seed437 (GPU0, 8 worlds) reached update 200 / 48,240 segment transitions (rounded 2 FPS; +1,440 in 10 minutes), rolling mean -24.8. Its durable `training_state.pt` advanced at 17:34:52 IST to the update-200 boundary. Add continuation offsets 10,320 (436) and 14,640 (437): cumulative log-visible progress is 86,640 and 62,880 of 250,320. No new retry record or evaluation request was found. Free VRAM: GPU0 92,723 MiB; GPU1 90,660 MiB. No validation/test runs; held-out v5 test remains sealed. Next sample no earlier than 12:15:00Z.



## 2026-11-27T12:17:27Z consolidated recovery health sample

Both main learners remain alive. Seed436 (GPU1, 16 worlds) reached update 164 / 79,200 recovery-segment transitions (rounded 3 FPS), rolling reward mean -25.7; latest durable state/checkpoint is update 160, saved 12:08:08Z. Seed437 (GPU0, 8 worlds) reached update 207 / 49,920 segment transitions (rounded 2 FPS), rolling reward mean -24.8; latest durable state is update 200, saved 12:04:52Z. Add continuation offsets 10,320 (436) and 14,640 (437): cumulative log-visible progress is 89,520 and 64,560 of 250,320. No new retry files or evaluation requests; retry totals remain four (436) and zero (437). Free VRAM: GPU0 92,723 MiB; GPU1 90,660 MiB. v5 validation/test remain unrun and held-out test sealed. No process or workload was changed. Next consolidated sample no earlier than 12:27:27Z.


## 2026-11-27T12:27:45Z consolidated recovery health sample

Both main learners remain alive at elapsed approximately 6h05m. Seed436 (GPU1, 16 worlds) reached update 169 / 81,600 recovery-segment transitions (rounded 3 FPS); latest reported rolling reward mean -25.5. Seed437 (GPU0, 8 worlds) reached update 214 / 51,600 segment transitions (rounded 2 FPS); latest reported rolling reward mean -25.3. Add continuation offsets 10,320 (436) and 14,640 (437): cumulative log-visible progress is 91,920 and 66,240 of 250,320. Durable state remains at update 160 for 436 (saved 12:08:08Z) and update 200 for 437 (saved 12:04:52Z); later updates are not checkpointed. Both PIDs 2042361/2042362 remain alive. No new retries or evaluation requests observed. Free VRAM: GPU0 92,723 MiB; GPU1 90,660 MiB. Held-out v5 test remains sealed. No process or workload was changed. Next sample no earlier than 12:37:45Z.

## 2026-11-27T12:27:45Z source sync

Committed I-019 implementation as `04d9458` and pushed feature branch `codex/waffle-gazebo-mission-20261126` from Kakarot to `rl-demo` GitHub. Kakarot, H100, and GitHub branch tips all equal `04d9458f8d07ba1afd00e96a2f0465a86e68aae2`; worktrees clean. `main` unchanged.


## Gate audit snapshot (2026-11-27T12:36:45Z; current evidence only)

- A, native Gazebo and sensor/physics qualification: PARTIAL. The official Waffle model worker has run actual Gazebo rollouts and the evaluator has completed a three-case validation evaluation; however, the profile remains illustrative and a six-case policy screen had 4/6 successes, one collision and one timeout. Full reference/optimized worker equivalence and renderer/transport qualification are not complete.
- B, perception/GST/actor/PPO contract: PARTIAL. Real Gazebo + DR-SPAAM + GST + actor/PPO path was exercised in a stage-logged smoke and active PPO training. Full fresh-process/recovery, temporal detector parity, no-leak audit and complete fault qualification remain incomplete.
- C, profile/scenarios: PARTIAL. v5 train/validation/test banks each contain 500 geometrically validated four-human scenarios; they are explicitly illustrative and contain no static obstacles. Physical calibration records and a surveyed lab map are absent.
- D, end-to-end smoke/recovery: PARTIAL. CPU fake-worker trace/quota fixtures and actual native training/recovery exist; full CPU/GPU/DDP smoke ladder, fault recovery and equivalence evidence remain incomplete.
- E, performance/placement: PARTIAL. Four-world eager vs scan/masked-GRU run measured about 10.05 vs 10.03 transitions/s (rollout dominated; no speedup). A 32-world probe produced a recovered worker SIGSEGV; no validated stable winner or matched single-vs-DDP comparison exists.
- F, convergence: IN PROGRESS / NOT PASSED. Seeds 436/437 are active on v5 with slow throughput; no 500-case validation checkpoint selection or predeclared quality target streak.
- G, paper metrics: PARTIAL. Paper-text ITR/SD online and independent offline scoring now have fixture coverage; trace schema now carries exact Gazebo timestamps and stable `person_i` IDs. Real saved trace production and full required aggregation/uncertainty checks remain pending.
- H, locked evaluation/handoff: NOT STARTED. No validation lock exists; the held-out test remains sealed. No final checkpoint, full evaluation report, plots/videos, manifest or deployment bundle has been produced.

The current source commit is `927f7be1e2a0a71e29086dceb244e3e717f53f38`, pushed to the feature branch on GitHub and synchronized to Kakarot and H100. This audit is not a completion claim; next action is continue the active v5 learners, monitor at the recorded 600-second interval, and address independent gaps without changing their profile/checkpoint contract.


## 2026-11-27T12:52:49Z consolidated recovery health sample

Both recovery learners remain alive after approximately 6h30m. Seed436 (GPU1, 16 worlds) reached update 181 / 87,360 recovery-segment transitions (rounded 3 FPS; +3,360 since the 12:39Z sample), latest rolling reward mean -25.3. Its durable state/checkpoint advanced to update 180 at 12:50:27Z. Seed437 (GPU0, 8 worlds) reached update 229 / 55,200 transitions (rounded 2 FPS; +1,920 since the 12:39Z sample), latest rolling reward mean -24.9; durable state/checkpoint remains update 220, saved 12:37:11Z. Add continuation offsets 10,320 (436) and 14,640 (437): cumulative log-visible progress is 97,680 and 69,840. Both PIDs 2042361/2042362 are alive. Free VRAM remains 92,723 MiB GPU0 and 90,660 MiB GPU1; host memory available 463 GiB, swap used 468 MiB, /dev/shm free 252 GiB, filesystem free 7.0 TiB. The queried active worker log roots contain no retry ledgers; this is not evidence beyond those roots, and historical retry counts remain four (436) and zero (437). No learner process was changed. Next health sample no earlier than 13:02:49Z.

## I-019 real Gazebo trace smoke and I-020 aggregate parity fix (2026-11-27T12:45Z)

Ran a one-case v5 validation-only evaluator smoke using seed436 durable checkpoint 00160, seed 190211, GPU1 (5% PyTorch allocation cap), actual Gazebo worker, detector/GST/policy and JSON scan transport. Exit code 0 at 12:45:44Z. Scenario `v5-validation-0489` ended in a collision after 8 policy steps; this single case is diagnostic, not candidate selection or policy qualification. Its saved trace was independently recomputed: one trace, ITR 0.625 (5/8 steps), pooled SD 0.9852528313388976 m, and online/offline parity passed. A prior 15-step validation smoke exposed an aggregate comparison failure even though per-episode trace parity passed: online float32 aggregate rounding exceeded the evaluator's 1e-12 tolerance. Root cause was the evaluator comparing float32 online values as if float64-exact. The fix uses a 1e-6 parity bound, reports exact float64 recomputed trace metrics, and enforces unique scenario trace matching. The initial SHM attempt separately failed before a worker launched because the Podman backend explicitly supports JSON only; retrying with JSON transport was the proper configured path. Focused fake-worker/scorer tests passed 11 tests; py_compile and diff checks passed. Test split remains untouched.

## 2026-11-27T12:52Z source sync state

Current H100 feature tip is `a6df10c378f43e61223180695831106811a87cf2`, clean and one commit ahead of GitHub's feature ref `927f7be`. Kakarot is still clean at `927f7be`. Kakarot-to-H100 fetch is waiting on a Tailscale additional-check challenge; no source changes were made locally and the new commit is not yet pushed. The earlier `927f7be` commit remains synchronized. `main` unchanged.


## 2026-11-27T12:57:55Z source sync complete

Transferred commit `a6df10c378f43e61223180695831106811a87cf2` as a verified 2.7 KiB Git bundle over the already-open H100 session after a new Kakarot-to-H100 SSH connection paused for Tailscale additional-check. Kakarot fast-forwarded the feature branch and pushed it to GitHub; H100 fetched the GitHub branch. Kakarot, H100 and GitHub all verify the same commit, worktrees clean, `main` unchanged. Bundle SHA256: `63b99db4290119d82292f915de9a4bf7f4ef7f8f533c2f8a6897a48578b0dd8a`. No credentials or run artifacts were included in the bundle.


## 2026-11-27T13:08Z timestamp coverage validation and source synchronization

Committed `a4b563750ff14bcc2118f39f6f68f9203a524ac2` (`Validate simulator timestamp coverage in paper traces`). The independent trace scorer now requires the recorded policy interval, monotonically increasing timestamps at that interval, aligned robot/human timestamps, and timestamp coverage through each complete future horizon. A malformed-timestamp regression was added. The focused quota, trace integration, and paper-metric test set passed 12 tests; syntax compilation and `git diff --check` passed. The saved real Gazebo validation trace was rescored successfully (ITR 0.625; pooled SD 0.9852528313388976 m; online/offline parity true). This remains a one-case validation diagnostic, not checkpoint qualification; the held-out test split remains untouched.

Source commit `a4b563750ff14bcc2118f39f6f68f9203a524ac2` is verified on Kakarot, GitHub feature branch, and H100; all three worktrees are clean and `main` is unchanged. Temporary transfer bundles were removed.


## 2026-11-27T13:12Z native simulator implementation review

Completed a full-file semantic read of `gazebo_sim/vector_env.py`, `worker.py`, `world.py`, `worker.cc`, and `hardware.py` (1,221 source lines combined). The detailed coverage and limits are recorded in CODE_AUDIT_PROGRESS.md. No new defect was confirmed and no source/run process changed. Simulation fidelity, complete repository review, qualification, convergence, locked evaluation, and handoff remain outstanding.


## 2026-11-27T13:13:43Z consolidated recovery health sample

Both recovery learners remain alive: seed436 PID 2042361 on GPU1 (16 worlds), and seed437 PID 2042362 on GPU0 (8 worlds). Latest progress rows: seed436 update 191 / 92,160 recovery-segment transitions, 3 reported FPS, rolling episode reward mean -24.30; seed437 update 243 / 58,560 segment transitions, 2 reported FPS, reward mean -25.41. From the prior 12:52:49Z sample this is +10 updates/+4,800 transitions for seed436 and +14 updates/+3,360 transitions for seed437. Adding recorded continuation offsets 10,320 and 14,640 gives cumulative log-visible transitions 102,480 and 73,200. Latest durable checkpoints: seed436 update 180 (12:50:27Z; no newer durable state), seed437 update 240 (13:09:45Z). These counters are segment/log-visible counters, not verified unique environment transitions across recovery.

Free VRAM: GPU0 92,723/95,830 MiB, GPU1 90,660/95,830 MiB. Host available memory 463 GiB; swap used 468 MiB; /dev/shm free 252 GiB; root filesystem free 7.0 TiB. Both PIDs and live worker episode files were present. No startup retry ledger was found in the two queried worker roots; this statement is limited to those paths. No learner was changed. Next stable health sample no earlier than 13:23:43Z.


## 2026-11-27T13:14Z correction to retry-ledger search scope

Correction to the 13:13:43Z sample entry: the first glob used a pattern that did not match the nested worker directory layout, so its statement that no retry ledger was found was incorrect. A recursive scan found five retry records under the active seed436 log root and one under seed437. Four seed436 worker exits and the seed437 exit report code 139; one seed436 record has no exit code. All records are retryable worker-closed-stdout failures on startup/reset, and both learner PIDs remain alive. This corrects the earlier root-level search result; the 13:13:43Z progress, checkpoint and VRAM values are unchanged.


## 2026-11-27T13:24:09Z consolidated recovery health sample

Both recovery learners remain alive: seed436 PID 2042361 on GPU1 (16 worlds), and seed437 PID 2042362 on GPU0 (8 worlds). Latest progress rows: seed436 update 195 / 94,080 recovery-segment transitions, 3 reported FPS, rolling reward mean -25.23; seed437 update 249 / 60,000 segment transitions, 2 reported FPS, reward mean -25.37. From the 13:13:43Z sample, seed436 advanced +4 updates/+1,920 segment transitions and seed437 +6/+1,440. Adding the recorded continuation offsets 10,320 and 14,640 gives cumulative log-visible transitions 104,400 and 74,640. Latest durable checkpoints remain seed436 update 180 (12:50:27Z) and seed437 update 240 (13:09:45Z). These are segment/log-visible counters, not verified unique environment transitions across recovery.

Free VRAM: GPU0 92,723/95,830 MiB; GPU1 90,660/95,830 MiB. Host available 463 GiB; swap used 468 MiB; /dev/shm free 252 GiB; root filesystem free 7.0 TiB. Recursive retry-ledger scan finds five seed436 and two seed437 worker closures (six code 139 exits, one unavailable exit code). Both learners and active Gazebo episode logs remain present; neither process changed. Next stable sample no earlier than 13:34:09Z.


## 2026-11-27T13:26Z mission ledger archive synchronization

The updated status, issue register, source audit coverage, structured state and append-only command ledger were added to the feature branch as commit `20449b109a26538da5f61fd15766bb26af520351`. Kakarot fast-forwarded and pushed this branch to GitHub; H100 fetched it. All three checkouts verified the same commit with clean worktrees and `main` unchanged. Per-file SHA256 values are in the tracked SYNC_MANIFEST.json.


## 2026-11-27T13:32:59Z 500-case validation diagnostic launched

A first launch of `eval_v5_bank_checkpoint.sh` passed its split/checkpoint admission gate but failed before starting Gazebo because the embedded Python `sys.argv` list ended with a parenthesis instead of a closing bracket. No workers or output directory were created; the failed shell log is preserved at `logs/eval-seed437-update240-validation500.log`, and its stale request marker was removed. Fixed the bracket and confirmed shell syntax plus Python compilation of the exact embedded block. The rerun passed admission for exactly 500 validation cases with seed437 update240 checkpoint SHA256 `9fba327ce8ca22e198306abe817de390742957b3be3fcea9cfd032d0e28ce914`. Evaluation PID 3386551 / Python PID 3386575 is active on GPU1. At the 13:32:59Z check, GPU1 had 88,726/95,830 MiB free and two per-episode trace files had been emitted; evaluation is incomplete. This is diagnostic validation only and cannot select a final checkpoint by itself. The held-out test split remains untouched.


## 2026-11-27T13:35:53Z recovery and validation progress

Seed436 PID 2042361 (GPU1, 16 worlds) reached update 200 / 96,480 recovery-segment transitions, 3 FPS, rolling reward -25.54; durable update200 saved 13:33:40Z. Seed437 PID 2042362 (GPU0, 8 worlds) reached update256 / 61,680 transitions, 2 FPS, reward -25.57; durable state remains update240 from 13:09:45Z. Since 13:24:09Z, segment counters advanced +2,400 (436) and +1,680 (437); with continuation offsets, cumulative log-visible progress is 106,800 and 76,320. Both learners remain alive; retry record counts remain 5 and 2.

The validation-only seed437 update240 evaluation is active (bash 3386551, Python 3386575), with 61 of 500 episode trace files present and no aggregate metrics JSON yet. GPU0 free VRAM 92,723 MiB; GPU1 88,412 MiB. The held-out test split remains untouched. Next learner health sample no earlier than 13:45:53Z.


## 2026-11-27T13:46:11Z recovery learners and validation progress

Seed436 PID 2042361 (GPU1, 16 worlds) is at update 205 / 98,880 segment transitions, 3 reported FPS, reward mean -25.74; durable update200 saved 13:33:40Z. Seed437 PID 2042362 (GPU0, 8 worlds) is at update262 / 63,120 transitions, 2 FPS, reward mean -25.15; durable update260 saved 13:41:57Z. Since 13:35:53Z, counters advanced +5/+2,400 (436) and +6/+1,440 (437); adding continuation offsets gives 109,200 and 77,760 cumulative log-visible transitions. Retry ledgers remain 5 seed436 and 2 seed437.

The seed437 update240 500-case validation diagnostic remains active (PID 3386551 and child 3386575), with 247 trace files and no metrics JSON yet. Free VRAM: GPU0 92,723 MiB; GPU1 88,412 MiB. No learner changed. The held-out test split remains untouched. Next health sample no earlier than 13:56:11Z.


## 2026-11-27T13:54Z audit line-count correction

Correction: `gazebo_sim/worker.cc` is 354 lines, not 333. The full-file read covered all 354 lines; the earlier claimed 1,221-line combined total was incorrect and is withdrawn.


## 2026-11-27T13:56:44Z recovery and validation progress

Seed436 PID 2042361 (GPU1, 16 worlds) reached update210 / 101,280 recovery-segment transitions, 3 reported FPS, rolling reward -24.46; durable update200 remains at 13:33:40Z. Seed437 PID2042362 (GPU0, 8 worlds) reached update268 /64,560 transitions, 2 FPS, reward -25.45; durable update260 remains at 13:41:57Z. Since 13:46:11Z, progress is +5 updates/+2,400 transitions (436) and +6/+1,440 (437); cumulative log-visible transitions with continuation offsets are 111,600 and 79,200. Retry counts remain 5 and 2.

The seed437 update240 500-case validation diagnostic (PIDs 3386551/3386575) has 447 episode traces; aggregate metrics are not yet written. Free VRAM: GPU0 92,723 MiB; GPU1 88,306 MiB. Both training learners were preserved. Test split remains untouched. Next learner health sample no earlier than 14:06:44Z.


## 2026-11-27T14:07:14Z seed437 update240 500-case validation completed

The corrected guarded evaluator completed all 500 unique v5 validation cases in 1,648.9s (3,196 counted transitions, 2.212 transitions/s); no test case was read. Outcomes: 0 successes, 500 collisions, 0 timeouts (95% Wilson success interval 0 to 0.76%). Mean reward -25.139; mean episode duration 1.598s; mean path 0.00138m; mean gate-intervention fraction 0.587 and gate acted in 94.8% of episodes. Mean future-horizon ITR is 0.811945; pooled current-time intrusion SD 0.779291m over 2,098 intrusion steps. Offline recomputation passed for all 500 traces with exact aggregate parity. This checkpoint is a clear validation failure, not a candidate; the near-zero path and all-collision outcomes indicate the deterministic policy is effectively stationary in this bank. No validation lock or final test was created. Full metrics, admission receipt, execution log and 500 compressed trace files are archived in the repository with per-file SHA256 manifest.

Independent seed436 checkpoint update200 validation is active on GPU0 (PID3888193); 42 of 500 traces were present at 14:07:14Z. GPU0 free VRAM 90,478 MiB; GPU1 90,660 MiB. Existing learners were left running. Next health sample no earlier than 14:17:14Z.


## 2026-11-27T15:30:38Z independent seed436 validation and action/gate diagnosis

The seed436 update200 checkpoint (SHA-256 `9c7967c1087239af6643484cae3bc2f524c9e12a47e07cfe9f2d63e340f5b380`) completed all 500 unique validation cases in 1,628.1s (3,048 transitions; 2.024 transitions/s). Outcomes were 0 successes, 500 collisions, 0 timeouts; 95% Wilson success interval 0 to 0.76%. Mean reward -24.783, path 0.05469m, duration 1.524s, goal distance 1.159m, gate acted in 97.6% of episodes and 58.8% of policy steps. Mean future-horizon ITR 0.81027, pooled intrusion SD 0.77978m over 2,069 intrusion steps. Offline recomputation passed for all 500 traces. The held-out test remains untouched; no checkpoint is selected.

A reproducible one-case validation diagnostic on seed437 update300 checkpoint reproduced scenario `v5-validation-0085` and a collision after 9 transitions. The gate blocked 34/45 physics substeps; 5 substeps had sensor/tracker faults and 29/40 non-fault substeps were blocked with fresh scans. During the first five policy decisions it tracked four people and scan ages were 0.05-0.20s; later, one 5-substep decision had zero tracked humans. This identifies both gate intervention and a brief perception fault in one episode, not a complete explanation for the 500-case failures. Artifacts and the reproducible harness are archived.

Consolidated H100 sample at 15:30:38Z: seed436 PID 2042361 alive at update253 / 121,920 segment transitions (3 reported FPS; latest rolling reward -25.120); seed437 PID 2042362 alive at update326 / 78,480 (2 reported FPS; latest rolling reward -24.678). Latest durable checkpoints remain update240 for seed436 and update300 for seed437. GPU free VRAM: GPU0 92,703MiB (0% utilization), GPU1 90,660MiB (90%); host available 463GiB, swap 468MiB, root filesystem free 7.0TiB. Both learners preserved. Next stable health poll no earlier than 15:40:38Z.


## 2026-11-27T15:46:35Z publication sync and recovery health

Archive commit `8bcaf346a4589e2b21466d1d3cdbfe62074724c5` is present on the H100 branch, fast-forwarded to Kakarot, and pushed through Kakarot to GitHub branch `codex/waffle-gazebo-mission-20261126`. The GitHub branch head, Kakarot HEAD and H100 HEAD were all verified equal at that commit; both worktrees were clean. Kakarot rechecked the root sync manifest and all four evaluation manifests: 503 artifacts in each 500-case evaluation, 7 and 8 artifacts in the two one-case diagnostics, with no hash or size mismatches. The mission record mirrors the failed validations, all traces, and the diagnostic harness; no test split was read.

Consolidated H100 sample at 15:46:35Z: seed436 PID 2042361 alive at update260 / 125,280 segment transitions (3 reported FPS; latest rolling reward -25.088); seed437 PID 2042362 alive at update335 / 80,640 (2 reported FPS; latest rolling reward -24.925). Latest durable checkpoints are seed436 update260 and seed437 update320. GPU free VRAM: GPU0 92,703MiB (0% utilization), GPU1 90,660MiB (92%); host available 462GiB, swap 468MiB, root filesystem free 7.0TiB. Both learners preserved. Next stable health poll no earlier than 15:56:35Z.


## 2026-11-27T16:14:24Z seed437 update340 action/gate diagnostic

The 100-episode, 100-unique-scenario validation diagnostic completed on H100 GPU0 in 311.8s with exit status 0. Results: 0 successes, 100 collisions, 0 timeouts; mean reward -24.694, mean path 0.001010m, mean duration 1.5125s, and final goal distance 1.2279m. Offline paper-trace verification passed for all 100 traces.

Action analysis found all 605 latent forward components below zero (mean -0.502817, range [-0.583763,-0.313935]). The native forward-only mapping projected all of them to zero linear speed in every guard substep. Across 3,025 guard substeps, 1,702 (56.3%) were blocked; sensor/tracker faults were present in 85 (2.8%), and scans were not stale beyond one second. Initial surface clearance ranged from 0.0813m to 1.2292m; 43 scenarios started above 0.4m but still had no successes and averaged 0.001358m path. The evidence points to collapsed forward-action output as the dominant observed failure for this checkpoint. The gate remains intact; no test cases were read and the active learners were preserved.

Full archive with metrics, eight action logs, 100 compressed traces, 116 Gazebo worker logs, run receipt, analysis, and per-file SHA-256 manifest is at `evaluations/seed437_update340_validation_action_gate100_20261127/`.


## 2026-11-27T16:35:41Z matched sigmoid-forward counterfactual

A diagnostic-only sigmoid forward mapping was evaluated on the same 100 validation scenarios and checkpoint as the checkpoint-native action/gate run; the safety gate remained unchanged. All 100 episodes still collided, with 0 successes and 0 timeouts. Mean path increased from 0.001010m to 0.021136m; mean duration was 1.5225s. The gate was active in 97% of episodes and blocked 1,751/3,045 physics substeps (57.5%); sensor/tracker faults accounted for 85/3,045 (2.8%). Scenarios matched one-to-one and offline trace verification passed. The action-map dead zone explains near-zero movement but changing the mapping alone did not solve the navigation failure. This counterfactual is not a qualified checkpoint or action contract.

Full 229-file archive with metrics, action logs, 100 traces, worker logs, receipt, analysis, and per-file SHA-256 manifest: `evaluations/seed437_update340_validation_action_gate100_sigmoid_forward_counterfactual_20261127/`.


## 2026-11-27T16:44:31Z seed437 update360 matched validation sample

The durable update360 checkpoint completed the same 100 validation scenarios as update340 using the checkpoint-native action mapping. It again produced 0 successes, 100 collisions, 0 timeouts; mean path was 0.000994m, duration 1.5125s, and final goal distance 1.2279m. All 605 deterministic latent forward outputs were negative, so all 3,025 forward requests were zero. The gate blocked 1,678/3,025 substeps (55.5%) and had activity in 95% of episodes; 80/3,025 substeps (2.6%) had sensor/tracker faults. Paper-trace offline verification passed. The matched update340 sample had the same outcomes and practically identical path, so 20 additional updates did not recover deterministic movement.

The 229-file evidence archive is `evaluations/seed437_update360_validation_action_gate100_20261127/`. The full-bank test split remains untouched; active training processes remain unchanged.

## 2026-11-27T17:54Z contact-source probe and archive correction

A fresh one-episode validation diagnostic using seed436 update280, case `v5-validation-0225`, exercised new non-invasive telemetry in `diagnose_v5_action_gate.py`. It completed five scored policy decisions. The Gazebo worker's `contact` flag was false on all scored policy substeps; the evaluator terminated as Collision because conservative robot-human disk clearance reached -0.008262 m, while wall clearance stayed positive. During the four unscored human-only paper-lookahead steps, the worker flag was true on 14 substeps as humans moved around the stationary terminal robot. The worker flag combines external contact with a tilt event, so it is labeled contact-or-tilt; this sample establishes a proxy-triggered terminal without an observed worker flag during the scored policy interval, not fleet-wide physical-contact rates. Artifacts are under the H100 run's `evaluations/seed436_update280_contactprobe1_validation_20261127/` and its exit receipt reports status 0. No test case or learner was changed.

Review found the first seed436 update280 archive omitted its eight action/gate JSONL files. They are now included; SHA256SUMS covers 112 archive files and verifies. Public archive copies of `metrics.json` and `launcher.log` redact the machine-specific run-root prefix; original files remain in the H100 run directory. The diagnostic harness now captures worker contact-or-tilt flags separately for policy substeps and post-terminal lookahead. Syntax compilation and the one-case actual-Gazebo diagnostic passed. The full matched 100-case contact-source diagnostic remains pending.

Consolidated H100 sample at 2026-11-27T17:54:48Z: seed437 PID 2042362 alive at update 412 / 99,120 transitions; seed436 PID 2042361 alive at update 320 / 154,080 transitions. Latest durable checkpoints are seed437 update400 and seed436 update320. Free VRAM: GPU0 92,701 MiB; GPU1 90,660 MiB. Rolling rewards remain around -25.0 and -25.3 respectively; training is active but not converged. No learner was changed. This is the latest consolidated health-check timestamp; next stable check no earlier than 18:04:48Z.

## 2026-11-27T18:09Z contact attribution and pretrained-transfer audit

The matched 100-case seed436 update280 validation telemetry run completed at 18:02:39Z (exit 0). It reproduced the previous result exactly: 0 successes, 100 evaluator Collision labels, 0 timeouts; 611 policy steps; mean path 0.059617m, duration 1.5275s, gate intervention fraction 0.58756, mean paper ITR 82.2969%, pooled SD 0.778232m; offline trace verification passed. Source attribution: all 100 terminal rows had negative human disk clearance, no wall disk overlap; 1/100 had a Gazebo worker contact-or-tilt flag during scored policy steps, so 99/100 were proxy-only human-disk terminals without that worker flag. The worker flag combines physical contact and tilt, so the single event is not proven contact. 696 worker contact-or-tilt flags occurred during the unscored post-terminal human-only lookahead. Archive: `evaluations/seed436_update280_contact_source_validation100_20261127/` (metrics, 8 action logs, 100 truth traces, 116 worker files, launcher and exit receipt; SHA256 manifest verified). Test split untouched.

A training-entrypoint audit found both active v5 seed lineages are scratch actor-critic runs, not pretrained actor transfer. Seed436 recovery state records `resume=false`, `waffle_init_cartesian=false`, and `continue_state` from its own scratch run; the original launcher omitted `--resume`, `--load-path`, and `--waffle-init-cartesian`. `train.py` loads `load_path` only when `resume=true`, so the default non-random `41200.pt` path in run_args was not loaded. Seed437 followed the same launcher pattern. Their records remain valid scratch controls; no learner was stopped or changed. I-026 is open. Next experiment will use the official randomized `41665.pt` with explicit `--resume --waffle-init-cartesian` and a fresh optimizer, in a new run directory.

Consolidated H100 health sample at 2026-11-27T18:09:03Z: seed437 PID 2042362 alive at update421 / 101,280 transitions, latest checkpoint update420; seed436 PID 2042361 alive at update327 / 157,440 transitions, latest checkpoint update320. Progress remains poor (rolling rewards about -25.07 and -24.71); neither scratch run is converged. Free VRAM: GPU0 92,701MiB; GPU1 90,660MiB. Both runs preserved. Next stable poll no earlier than 18:19:03Z.


## 2026-11-27T18:20:54Z consolidated health check and pretrained transfer smoke

At the permitted consolidated poll, seed437 PID 2042362 was alive at update 428 / 102,960 transitions (latest durable checkpoint 00420.pt; latest progress reward -24.86) and seed436 PID 2042361 was alive at update 332 / 159,840 transitions (latest durable checkpoint 00320.pt; latest progress reward -25.35). Both remain scratch controls and are not converged. Free H100 memory was 92,701MiB on GPU0 and 90,660MiB on GPU1; utilization was 0% and 91%. An additional 564MiB compute process on GPU1 was left untouched. Next stable health poll is not before 18:30:54Z.

A separate seed438 transfer smoke loaded official randomized actor `41665.pt` (SHA256 `a60bdd3b...0cfd8ce`) with `--resume --waffle-init-cartesian --waffle-native-action`, a fresh PPO optimizer and no continuation state. Actual Gazebo Harmonic and DR-SPAAM ran on GPU0 for one PPO update and 60 segment transitions (8 episode rewards, mean -22.7), exit 0. Native output checkpoint hash is `a690bd8c...fc6db6`; recoverable optimizer-state hash is `6b5a2f35...5b58b`. A fresh evaluator loaded that native checkpoint and ran one validation case, `v5-validation-0393`; the case collided after 8 policy steps (path 0.05631m, 2.0s, minimum human clearance -0.05649m). Offline paper-trace verification passed. This confirms transfer and evaluator wiring only; the observed case failed and the smoke does not establish policy quality or convergence. The first evaluator attempt stopped at the CLI process/minibatch divisibility guard before creating Gazebo; the corrected call used one minibatch. Test split was not used. Full artifacts and hashes are in `evaluations/pretrained_transfer_smoke_seed438_20261127/`.


## 2026-11-27T18:31:26Z consolidated health check; sustained pretrained training launched

At the global poll, seed437 PID 2042362 was alive at update 434 / 104,400 transitions (last durable checkpoint 00420.pt; last reward -25.55), and seed436 PID 2042361 was alive at update 337 / 162,240 transitions (last durable checkpoint 00320.pt; last reward -24.91). Both remain slow, low-reward scratch controls; neither is converged. GPU free memory was 92,701MiB and 90,660MiB; utilization 0% and 92%. The separate PID 444191 on GPU1 was preserved. Next health check is not before 18:41:26Z.

Started the distinct pretrained seed438 target run at 18:31:36Z on GPU0: launcher PID 3235188, learner PID 3235199. It confirmed strict load of randomized actor 41665.pt (SHA256 `a60bdd3b...0cfd8ce`) and uses `--resume --waffle-init-cartesian`, native actions, a fresh PPO optimizer and no continuation state. Configuration is 32 actual Gazebo worlds, 30-step recurrent rollouts, 8 PPO minibatches, 5 epochs, train split, and 250,000 requested transitions (260 updates / 249,600 effective transitions due rollout divisibility). Policy and full optimizer state are saved every 10 updates. Durable log and exit receipt are under `$RUN_ROOT/logs/`; output/checkpoints under `$RUN_ROOT/experiments/`. This is a new, separate lineage; scratch runs remain untouched. The 32-world placement was selected from the existing two-update H100 screen (best measured among 4/32 worlds, 4.706 transitions/s and 2.61GiB allocator peak); this is only a short screen, not sustained-throughput qualification.


## 2026-11-27T18:36Z paper-metric and case-schedule audit

Static review confirmed the paper metric implementation and offline trace verifier contracts. A formula-level enumeration for 500 scenarios across lane counts 1 through 64 found no duplicate or omitted indices under current balanced quotas. The targeted paper-metric unit suite passed (13 tests, 0.011s). This does not replace an actual full unique 500-case Gazebo evaluation; the original paper's SD convention remains explicitly documented as an interpretation. Test split untouched.


## 2026-11-27T18:38Z paper-regime scenario gap found

Read the locally stored primary paper text alongside the v5 generator and all six current profile inventories. The paper's randomized regime has within-episode random goal changes and per-human maximum speed 0.5–1.5m/s/radius 0.3–0.5m; the nonrandomized regime uses common human speed/radius and no random goal changes. The current v5 bank has 500 train/validation/test cases, all with four humans; each human follows a fixed cyclic two-waypoint path, with heterogeneous per-case speed 0.55–0.95m/s and radius 0.26–0.32m. It implements neither paper regime exactly, and its `mission_note` is stale (still says v3). I-028 is open. The running seed438 line is reclassified as exploratory nominal training/throughput evidence, not paper-regime-matched training. No test cases were executed or inspected for behavior.


## 2026-11-27T18:43:07Z consolidated health check

All three mission learners are alive and advancing. Seed437 scratch: update439 / 105,600 transitions, latest checkpoint00420, rolling reward -25.69. Seed436 scratch: update341 / 164,160 transitions, latest checkpoint00340, rolling reward -25.67. New seed438 32-world exploratory pretrained line: update2 / 2,880 transitions, first policy checkpoint `00000.pt`, reward -25.60; full learner-state save is scheduled at update10. It has progressed at about 4.17 global transitions/s across startup and the first updates. Free VRAM: 86,632MiB on GPU0 and 90,660MiB on GPU1; utilization 0% and 90%. Host available RAM 448GiB; root filesystem free about 7.0TiB. Separate PID444191 (564MiB on GPU1) left untouched. The seed438 line remains exploratory because v5 does not match either paper human regime. Next global poll no earlier than 18:53:07Z.

## 2026-11-27T19:34:23Z Paper human-regime implementation and H100 smokes

H100 source commit 829acd219cae4c7053314d70c29cd1453b9ba9af implements separate nominal randomized/nonrandomized human-dynamics profiles, hidden scheduled goal changes, goal-sampling diagnostics, and shutdown environment summaries. Kakarot and GitHub branch codex/waffle-gazebo-mission-20261126 were fast-forwarded and verified at the same commit. The source tree is clean on H100.

Both committed profiles passed validate_gazebo_profile and integrity checks: 500 unique cases in each of train/validation/test, four humans per case, and no duplicate scenario across or between regimes. Profile SHA256 randomized=57b373a1c6d63cc281c5699071414b3733dd4084aca8ccd43717ce43fcb68931; nonrandomized=c6cd247d1f9acd3aee76a958e0a2a463c8cbd099b2d289c46d0b2201754d3d39.

Real Gazebo Harmonic + DR-SPAAM pretrained PPO smoke runs completed on the train split, with no test data read. Randomized run paper_randomized_regime_smoke_r4_20261128 completed one PPO update / 20 transitions, started four episodes, and recorded zero goal changes and zero sampling failures; each episode reset before a five-second event could occur. Nonrandomized run paper_nonrandomized_regime_smoke_20261128 completed one PPO update / one transition, started one episode, and recorded zero changes/failures. These prove wiring and startup only, not convergence or navigation quality. The 5-second trigger is separately covered by a deterministic scheduler test and a PerWorld integration-hook test.

Focused evidence on the committed source: 8 human-dynamics tests, 3 scenario-bank tests, 1 broken-pipe cleanup test, 6 Gazebo environment integration tests, plus py_compile of train.py, worker.py, vector_env.py, human_dynamics.py and scenario generator; all passed. A preliminary launch without WAFFLE_GAZEBO_PODMAN_IMAGE failed before simulation because the host binary could not find libgz-sim8.so.8; a pre-created output directory also correctly tripped the fresh-output guard. Correct Podman environment and fresh run IDs produced the successful smokes. Worker cleanup now suppresses broken-pipe errors so startup failures retain their primary cause.

At the last consolidated training health sample (2026-11-27T19:20:47Z), seed437 PID2042362 was alive at update454/109200 segment transitions (latest checkpoint 00440); seed436 PID2042361 was alive at update353/169920 segment transitions (checkpoint 00340); pretrained seed438 PID3235199 was alive at update12/12480 transitions (checkpoint 00010). Seed438 remains an exploratory v5-profile run and is not a paper-regime result. Next global health sample no earlier than 2026-11-27T19:30:47Z.

Remaining gates include full-duration training on both new regimes, validation-selected candidates, 500-case locked evaluation per regime, full paper crowd-count coverage, hardware calibration and measured profile, benchmark qualification, final CI, and deployment handoff. No physical actuation was performed.


## 2026-11-27T19:38:59Z Record manifest reconciliation

The first integrity pass over the copied mission archive found two stale entries in its pre-existing root manifest: CODE_AUDIT_PROGRESS.md and scripts/diagnose_v5_action_gate.py in the tracked archive had later H100 additions than the corresponding run-root source copies. Both archive versions were preserved; I did not overwrite either. I refreshed manifest hashes from the actual tracked archive and recorded the discrepancy here. The state merge also retained all 132 keys from the prior tracked snapshot and merged in the 118-key current run-root snapshot plus this turn's progress fields.


## 2026-11-27T19:42:17Z consolidated learner health

The consolidated sample at 2026-11-27T19:41:07Z found all three existing learners alive: seed437 PID2042362 at update461/110880 segment transitions with checkpoint 00460; seed436 PID2042361 at update358/172320 segment transitions with checkpoint 00340; pretrained seed438 PID3235199 at update16/16320 transitions with checkpoint 00010. The v5 seed438 run remains exploratory. The next global health sample is no earlier than 2026-11-27T19:51:07Z.


## 2026-11-27T19:42:17Z archive reconciliation note

The tracked archive was newer than some run-root copies. Existing tracked sections were retained verbatim and unique run-root updates appended; the state snapshot was merged by key union. Previous and current versions of CODE_AUDIT_PROGRESS.md and scripts/diagnose_v5_action_gate.py remain preserved in their respective locations, with current tracked-archive hashes recorded in SYNC_MANIFEST.json.


## 2026-11-27T20:13:36Z paper-regime PPO startup

Two detached Gazebo PPO runs started from the official randomized actor with a fresh optimizer and native action head, using train splits only. Randomized run: seed 439 on GPU 0, 16 worlds; nonrandomized run: seed 440 on GPU 1, 16 worlds; each requests 250,000 transitions with 30-step recurrent rollouts, 4 mini-batches and 5 PPO epochs. The launcher runs repository train.py with normal on-demand CUDA allocation and no legacy 20% per-process cap.

At admission, free VRAM was 43,480 MiB on GPU 0 and 39,521 MiB on GPU 1; utilization was not queried. By 2026-11-27T20:13:36Z, each run had 16 world configs and 16 Podman worker launchers, with per-world episode logs appearing and the learner alive. The first PPO update was not yet observed, so this is startup evidence rather than a learning or quality result. Existing seed436, seed437 and seed438 jobs were left untouched. Next consolidated check is no earlier than 2026-11-27T20:23:36Z. Full hashes and artifact paths are in paper_regime_training_20261128.json.


## 2026-11-27T20:24:04Z first PPO update and consolidated health check

Both paper-regime learners remain alive and completed update index 0 after collecting 480 Gazebo transitions each. Randomized seed439 logged 95 episodes with mean reward -21.63 and median -20.6; nonrandomized seed440 logged 86 episodes with mean reward -23.42 and median -23.0. Each logged 1 FPS during this cold-start interval; this includes initialization and is not a steady-state throughput estimate. No policy-quality conclusion follows from one update. Free VRAM was 39,692 MiB on GPU0 and 35,476 MiB on GPU1. The three existing jobs also progressed: seed437 update474 / 114,000 transitions, seed436 update368 / 177,120, pretrained seed438 update26 / 25,920. All were alive and left untouched. Utilization was not queried. Next consolidated poll no earlier than 2026-11-27T20:34:04Z.

## 2026-11-28T05:03Z corrected paper-regime training admitted

The corrected randomized and nonrandomized profiles each completed one real Gazebo/DR-SPAAM/GST/PPO update in a one-world smoke. Randomized smoke seed443 recorded 2 scheduled goal changes, 0 sample failures, and 31 route arrivals; nonrandomized smoke seed444 recorded 0 random changes, 0 sample failures, and 40 route arrivals. Each smoke processed 30 transitions and exited 0. These are integration checks, not performance evidence. The smoke outputs and checkpoint hashes are in `paper_regime_training_20261128.json` and the H100 run root.

Fresh independent training is running from official actor41665 with a reset native action head and fresh PPO optimizer: randomized seed445 on GPU0 and nonrandomized seed446 on GPU1, 8 Gazebo worlds each, 8 minibatches, up to 250,000 transitions, train split. Admission snapshot before launch: GPU0 34,760 MiB free and GPU1 36,581 MiB free; utilization was not queried. The three legacy seed436/437/438 trainers remain untouched. Next consolidated status check is not before 2026-11-28T05:09:09Z.

## 2026-11-28T05:09:24Z first rollout collected

Both corrected 8-world trainers remain alive. Each collected 240 transitions (one 30-step rollout across eight worlds), but neither has logged its first PPO update. The rollout summaries report randomized seed445 52 episodes / mean reward -21.8 and nonrandomized seed446 39 / -23.2; these startup-only rewards do not establish policy quality. Each logged FPS 0 (integer rounded below one). GPU0 has 32,446 MiB free and GPU1 34,162 MiB free. GPU utilization was not queried. Episode log counts were 70 and 63. Legacy seed436/437/438 processes remain alive and untouched. Next consolidated check is not before 05:19:24Z.

## CI check - 2026-11-28T05:10Z

GitHub Actions query for `codex/waffle-gazebo-mission-20261126` returned 0 workflow runs. The two push workflows filter `main` and `waffle-pi-sim2real`; `workflow_dispatch` is available, but was not invoked. Therefore CI is unverified for this branch. Focused H100 regression tests passed: 11 human-dynamics tests and 3 scenario-bank tests.

## 2026-11-28T05:20:14Z consolidated training check

Both corrected trainers remain alive and advanced from update 0 / 240 transitions to update 2 / 720 transitions. The latest 100 reported episodes had mean reward -21.9 randomized and -23.8 nonrandomized, with logged FPS still rounded to 0; these are early-training diagnostics only. Current persisted checkpoints remain at update 0: randomized training_state SHA-256 `2ef655317546e3ca9e5346335ac81047ba12738ae5821ac616936eb983846773`, checkpoint `00000.pt` SHA-256 `d8089e1730aa08d43ec0ba8e4563cb1d08d32c3f7dc96fb74839621a3560c112`; nonrandomized training_state `a8b651e8aa26411bea536626c27a6e1edf46b6a353453ef2a31ab8ca04665092`, checkpoint `00000.pt` `0ed34f64e9980ee6c42b5426d3fed806e923748594674986baeda8d2b5dc35a8`. GPU0 free VRAM 32,426 MiB, GPU1 34,829 MiB; utilization not queried. Existing seed436/437/438 trainers remain alive and untouched. Next consolidated check no earlier than 05:30:14Z.

## 2026-11-28T05:31:55Z consolidated training check

Both new regime trainers remain alive at update 4 / 1,200 transitions each. Since the previous consolidated check each advanced 2 PPO updates and 480 transitions. Last-window mean rewards were -22.2 randomized and -23.1 nonrandomized; FPS remains below the logger's integer display resolution. These early summaries are not policy-quality evidence. Saved training states and checkpoints remain at update 0 with the existing hashes in the registry; next periodic save is pending. GPU0 free VRAM 32,376 MiB and GPU1 35,947 MiB; no utilization queried. Next check no earlier than 05:41:55Z.

## 2026-11-28T05:45:09Z consolidated training check

Both corrected runs remain alive at update 6 / 1,680 transitions each. The previous interval added 480 transitions per run over 794 seconds, an observed early rate of about 0.605 transitions/s per run. A linear projection to 250k transitions is roughly 114 hours per run; this includes cold startup, is not a reliable completion forecast, and says nothing about convergence. Last-window reward means were -22.0 randomized and -23.8 nonrandomized, with logged FPS rounded to 0. Persisted checkpoints remain at update 0. GPU0 has 31,939 MiB free and GPU1 38,042 MiB free; utilization was not queried. Existing runs 436/437/438 remain alive and untouched. Next check no earlier than 05:55:09Z.
