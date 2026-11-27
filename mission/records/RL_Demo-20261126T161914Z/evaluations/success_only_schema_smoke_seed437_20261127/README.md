# Success-only NT/PL evaluator smoke

Ran 2026-11-27 on H100 in Gazebo Harmonic, validation split only. Evaluator source revision: fcdd06740653ddd86cecfbb6b4482ea9f16d77ce. One scenario is a schema and integration smoke, not a quality estimate.

- Checkpoint: seed 437, update 360; SHA256 872e28688d12bf4cfd0b3ed201266c234160a4129eb7c86a0f3da473fe3e1d1c.
- Illustrative profile SHA256: 84557d48d8b92f589ccf62e2b719510ccfdaa83a27c970d42659d4cf94b494e7.
- Native Gazebo worker SHA256: 095521529c74cb135c7be804c21a374d7db4299a3928b3b47d0825479fe7e9aa.
- Runtime: CPU policy and CPU DR-SPAAM, 1 environment, 1 validation episode, seed 90437, four-step recurrent CLI shape. The first attempt stopped before simulation because num_processes=1 conflicted with the default minibatch count; retry set --num-mini-batch 1.

Result: scenario v5-validation-0157 collided (0/1 success). Success-only paper NT and PL are both JSON null, with successful episode count 0. The preserved all-outcome means remain 3.0 s and 0.0030062 m. Paper ITR was 5/12 = 0.416667; paper SD was 0.837269 m. Online and offline trace scoring agreed. This demonstrates field behavior on an actual Gazebo trace, not policy quality, hardware calibration, or final held-out evaluation.

Artifacts:
- metrics.json SHA256 0788c64cb119a28c4ba558a9b28ed9f47bf40c8d8501ad0b393f66fde15fa543
- metrics_traces/lane-000-episode-0001-case-000157.npz SHA256 a2d2f72800ab7568622bcacc45aaa5e715e9aed2460ab89163c41be583eaa779

The full external run directory is /home/rl-group4/waffle_runs/RL_Demo-20261126T161914Z/evaluations/success_only_schema_smoke_seed437_20261127.
