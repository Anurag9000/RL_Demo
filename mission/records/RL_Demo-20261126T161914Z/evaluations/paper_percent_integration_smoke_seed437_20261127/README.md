# Paper percentage field integration smoke

One-case actual Gazebo evaluator integration smoke on 2026-11-27, validation split only. Evaluator revision: f059c941571eb9b39e40047573dba6903d98d524. This checks output fields and trace parity; it is not a policy-quality or final-evaluation result.

- Candidate: seed 437, checkpoint update 360, SHA256 872e28688d12bf4cfd0b3ed201266c234160a4129eb7c86a0f3da473fe3e1d1c.
- Illustrative profile SHA256: 84557d48d8b92f589ccf62e2b719510ccfdaa83a27c970d42659d4cf94b494e7.
- Native Gazebo worker SHA256: 095521529c74cb135c7be804c21a374d7db4299a3928b3b47d0825479fe7e9aa.
- CPU policy and CPU detector, one process, validation case v5-validation-0157, seed 90437.

Outcome: collision (SR 0%, CR 100%, TR 0%). Since there were no successful episodes, paper NT and PL are null; all-outcome mean time/path remain separately present. Paper ITR is 41.6667%; paper SD is 0.837269 m. The online and independent offline trace calculations agree for the one trace. The case demonstrates schema behavior only.

Artifacts:
- metrics.json SHA256 317d2e14e1a95094a73fd604d39fd8b90f2e0cab31257c76a9d7a2a798471cee
- metrics_traces/lane-000-episode-0001-case-000157.npz SHA256 a2d2f72800ab7568622bcacc45aaa5e715e9aed2460ab89163c41be583eaa779

Full server run directory: /home/rl-group4/waffle_runs/RL_Demo-20261126T161914Z/evaluations/paper_percent_integration_smoke_seed437_20261127.
