# Seed437 update360 checkpoint-native action/gate validation

## Protocol

- H100 GPU0; 2026-11-27T16:38:23Z to 2026-11-27T16:43:40Z, exit status 0 (306.2s).
- Durable seed437 recovery checkpoint update360, SHA-256 `872e28688d12bf4cfd0b3ed201266c234160a4129eb7c86a0f3da473fe3e1d1c`; validation seed 900440; 100 unique scenarios and eight Gazebo lanes.
- Scenario IDs match the update340 baseline exactly. The held-out test split was not read and neither training learner was modified.
- Captured 605 policy-step action records, 3,025 guard substeps, 100 compressed paper-metric traces and 116 Gazebo worker log files. Offline trace verification passed.

## Result and update340 comparison

| Measure | Update340 | Update360 |
|---|---:|---:|
| Successes | 0/100 | 0/100 |
| Collisions | 100/100 | 100/100 |
| Timeouts | 0/100 | 0/100 |
| Mean path | 0.001010 m | 0.000994 m |
| Mean duration | 1.5125 s | 1.5125 s |
| Gate intervention fraction | 65.455% | 64.463% |

All 605 deterministic forward latent outputs were negative (mean -0.482654, range [-0.563617,-0.296867]), giving exactly zero native forward request in all 3025 substeps. The gate blocked 1678/3025 substeps (55.471%); sensor/tracker faults occurred in 80 substeps (2.645%). Mean path changed by -0.000016m from update340, with the same 100/100 collisions and same mean episode duration.

## Interpretation

Twenty more durable PPO updates did not change the deterministic forward-action collapse on this matched validation sample. The checkpoint is not a candidate. This remains evidence from one seed and a 100-case sample, not a full-bank qualification; no safety-gate change was made.
