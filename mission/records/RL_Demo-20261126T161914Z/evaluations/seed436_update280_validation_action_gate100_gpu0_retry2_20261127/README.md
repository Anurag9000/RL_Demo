# Seed 436 update 280 validation action and gate diagnostic

Completed on H100 at 2026-11-27T17:23:46Z (exit 0). This is a 100-case diagnostic on the validation split, not a final evaluation and not evidence of convergence. The test split was not used. Evaluator checkout was at revision `b7edf6baf0231d8e53244761cf7e75428f1b2d03`.

- Candidate: seed 436, update 280; checkpoint SHA256 `9e1be5871cf88e13c4aa0f12b3677a1c121c12d702eb3d63dd766df52b683004`.
- Episode seed: 90436; 100 unique validation scenarios; GPU 0; eight worker processes.
- Outcome: 0 successes, 100 collisions, 0 timeouts (SR 0%, CR 100%, TR 0%). Mean reward -24.5998, mean path 0.059617 m, mean duration 1.5275 s. Gate intervention fraction 0.58756; 96% of episodes had at least one intervention.
- Action log: 611 policy steps; latent forward output z0 mean 1.5873, median 1.6072, and negative on 0% of policy steps. Forward was requested on all policy steps. Across 3,055 safety-gate ticks, blocked fraction was 0.52799 and sensor-fault fraction 0.03110; accepted forward was nonzero on 47.2% of gate ticks. Nearest scan return median 0.6384 m, minimum 0.26365 m.
- Paper trace verification passed for 100 traces and 611 episode steps: mean episode ITR 82.2969%, pooled SD 0.778232 m over 423 intrusion steps.

Interpretation: this checkpoint requests forward motion consistently, unlike the earlier seed437 dead-forward sample, yet still collided in all 100 cases with very short paths and frequent gate intervention. The evidence shows the failure is broader than a dead forward clamp; it does not establish that the gate alone caused the failures. Preserve the safety gate while investigating policy behavior, gate deadlock, geometry, timing, and executed actions together. Do not select this checkpoint or treat this validation diagnostic as generalization evidence.

The first launcher attempt used a working directory outside the repository and failed relative GST loading. The next attempt could not reuse its nonempty output directory. Retry 2 used a fresh output directory and ran from the repository root; it completed successfully. The two failed-attempt logs remain in the external run folder.

Artifacts are listed in `SHA256SUMS`. Full external run directory: `/home/rl-group4/waffle_runs/RL_Demo-20261126T161914Z/evaluations/seed436_update280_validation_action_gate100_gpu0_retry2_20261127`.
