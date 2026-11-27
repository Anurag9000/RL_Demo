# Mission decisions

## D-001 — Separate physical contacts from conservative terminal proxies

The current Gazebo evaluator's `Collision` category combines a Gazebo worker contact-or-tilt flag, continuous robot-human disk overlap, and wall-disk overlap. The seed436 update280 100-case validation run has negative minimum human-disk clearance in all 100 episodes but did not archive per-substep contact attribution. A matched one-episode probe found zero worker contact-or-tilt flags during the five scored policy decisions and a human-disk-overlap terminal at -0.008262 m; contact-or-tilt flags appeared in the unscored human-only suffix. Therefore, do not call the 100 outcomes verified physical contacts. Preserve the disk safety proxy and gate for now. Add per-episode outcome-source telemetry and run the matched 100-case validation probe before changing collision termination, reward or safety logic. Any changed semantics require a new training lineage.

## D-002 — Reclassify the active v5 runs as scratch controls

The saved run args and original launcher establish that active seed436/437 runs did not transfer the paper actor: the initial commands omitted `--resume`, `--load-path` and `--waffle-init-cartesian`; `train.py` loads `load_path` only in the resume branch. `continue_state` restores those scratch runs' own model/optimizer state. Preserve both running jobs as scratch controls and keep their checkpoint genealogy; do not silently relabel or migrate them. Start a separate policy lineage from the official randomized actor checkpoint `trained_models/GST_predictor_rand/checkpoints/41665.pt`, explicitly load compatible weights, reset only the native action head, and create a fresh PPO optimizer.


## D-003 — Transfer smoke is a wiring gate, not a quality result

The separate seed438 smoke strictly loaded official randomized checkpoint 41665.pt, reset the native action head, used a fresh PPO optimizer and ran 60 actual Gazebo transitions. A fresh evaluator loaded the resulting native checkpoint and completed one validation trace, which collided. This proves only that the transfer/training/checkpoint/evaluation path executes. It does not show the pretrained policy has learned useful navigation or that the goal is met. Continue with a separate sustained pretrained run and use full validation evidence before selecting a candidate; keep test sealed.


## D-004 — Start sustained transfer at the measured 32-world screen configuration

The earlier H100 benchmark screen measured 4.706 transitions/s and 2.61GiB peak allocated memory at 32 Gazebo worlds, versus 1.805 transitions/s and 1.20GiB at 4 worlds; each was only two updates, so this is not sustained qualification. The separate official randomized pretrained transfer smoke passed model loading and produced a native checkpoint/optimizer state, while one validation episode failed. Start a distinct 250k-requested-transition (249,600 effective) transfer lineage at 32 worlds on GPU0, with a fresh optimizer and full-state saves every 10 updates, to gather meaningful learning evidence. Keep both prior scratch controls running and test sealed. Reassess throughput and learning on the next permitted global health check.


## D-005 — Keep the current v5 training run exploratory after the regime audit

The v5 profile provides 500 independent cases per split but its human dynamics do not match either primary-paper regime exactly: it has static cyclic two-waypoint paths and heterogeneous traits, no within-episode goal changes, and different speed/radius ranges. The 250k seed438 run has already started against this illustrative profile. Preserve it as an exploratory Gazebo/PPO and throughput baseline, but do not call it a paper-regime candidate or use it as final evidence. Build separate randomized and nonrandomized regime profiles/dynamics with explicit provenance before final regime evaluation; preserve the test split.
