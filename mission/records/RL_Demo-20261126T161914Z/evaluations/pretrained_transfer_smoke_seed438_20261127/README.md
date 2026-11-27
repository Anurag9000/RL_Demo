# Randomized pretrained transfer smoke

Run: `waffle_pretrained_transfer_smoke_seed438_rand41665_20261127`. Executed on H100 GPU 0 from code commit `c8f321b6df6c949e9766ff52b6197433d2432a20` on 2026-11-27. The input was the official randomized paper actor checkpoint `41665.pt` with the recorded SHA256. The launcher explicitly used `--resume --waffle-init-cartesian --waffle-native-action`, loaded the actor strictly, reset the native action head, and initialized a fresh PPO optimizer (no `--continue-state`).

The short real Gazebo Harmonic + DR-SPAAM training smoke used the illustrative v5 profile, four humans, two worker processes, seed 438, one PPO update and 60 segment transitions. Exit status was 0. Mean of the 8 reported training episodes was -22.7. The native output checkpoint and recoverable optimizer state are stored in the external run directory shown in `checkpoint_manifest.json`; hashes are included here.

A fresh evaluator process then loaded the output checkpoint and ran one validation case (`v5-validation-0393`), with offline paper-trace verification passing. Outcome was collision, path 0.05631m, duration 2.0s, 8 policy steps, minimum human disk clearance -0.05649m, ITR 62.5%. This is a checkpoint-load/evaluator smoke only; it is a single failing validation case, not evidence of policy quality or convergence. Test split was not used.

The first evaluator invocation stopped before Gazebo because `num_processes=1` conflicted with the default `num_mini_batch=2`; the corrected invocation set `--num-mini-batch 1`. The failed guard output is retained in `failed_eval_argument_guard.log`.

For reproduction, set the `WAFFLE_RUN_ROOT`, `WAFFLE_REPO_ROOT` and `WAFFLE_VENV` environment variables and run `scripts/run_pretrained_transfer_smoke.sh`. The validation command is captured in the mission command ledger.
