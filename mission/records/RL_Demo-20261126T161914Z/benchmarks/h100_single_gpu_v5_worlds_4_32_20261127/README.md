# Single-H100 Gazebo worker-count screen

These are short, single-GPU screening runs on the illustrative v5 Gazebo profile. Both measured only two PPO updates after warmup; they are not sustained benchmarks, parity tests, or convergence evidence. The run manifests do not include the source commit, so none is asserted here.

- Four environments: 240 global transitions, 1.805 transitions/s, 132.56 s rollout, 0.38 s learner, 1.20 GiB peak allocated / 1.23 GiB reserved.
- Thirty-two environments: 1,920 global transitions, 4.706 transitions/s, 406.92 s rollout, 1.10 s learner, 2.61 GiB peak allocated / 2.76 GiB reserved.
- Observed throughput ratio: 2.61x for the 32-environment two-update screen versus the four-environment two-update screen. Repeat sustained runs are needed before selecting a training configuration.
- Both used one process/rank on CUDA 0, four humans, train split, 30-step recurrent rollouts, 8 minibatches and 5 PPO epochs. Illustrative profile SHA256: 84557d48d8b92f589ccf62e2b719510ccfdaa83a27c970d42659d4cf94b494e7.
- The worker stage totals in benchmark.json sum across workers and overlap; reset_s, physics_ipc_s and detector_s are not additive wall time. Gazebo rollout collection dominated learner compute in both screens.

SHA256:
- worlds4/benchmark.json: b606c0a8c120bf46a082f894432827fff42402ca842ad93bb380f4cd29f30513
- worlds4/run_args.json: dacc19c7bc03cc2111c0f5f7fe008bd9e2ac0dc06ca3b5e8912ced09ed084536
- worlds32/benchmark.json: a38ff6a3aa89e53b8e960d7ac8ace2e20ad88ea7502d97c882d5d784f4c2eb55
- worlds32/run_args.json: fb8a33457e5abbd7442c688957129e22f1e0e52a967c196c320d258945018c2e
