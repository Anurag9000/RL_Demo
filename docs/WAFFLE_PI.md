# Waffle Pi experimental adaptation

Opt-in simulation path preserving the Cartesian checkpoint action semantics.
Use the original legacy dependencies in an isolated environment.

## Evaluate before retraining

From repository root:
```sh
python train.py --waffle-pi --waffle-eval-episodes 100 --load-path trained_models/GST_predictor_rand/checkpoints/41665.pt
```

## Fine-tune if needed

```sh
python train.py --waffle-pi --resume --freeze-human-attention --fast-training --load-path trained_models/GST_predictor_rand/checkpoints/41665.pt --output_dir trained_models/waffle_transfer --num-processes 32 --num-mini-batch 4 --num-env-steps 500000
```

Choose a new output directory. For full policy fine-tuning omit
--freeze-human-attention. GST weights stay frozen. Evaluate the resulting
checkpoint using the first command with the corresponding --load-path and
identical Waffle profile options. The original saved test.py configurations
are not the entry point for this profile.

The profile uses nominal wheel geometry, coupled saturation, a provisional
first-order motor model, 0.25 s policy steps, four pedestrians and a 90 s
timeout. It retains the existing circle-crossing human generator and original
pre-transition reward timing (including unicycle turn penalties); it is an
open arena, not a measured wall-aware lab model. Sensor range, footprint,
motor lag and acceleration must be calibrated. No physical motion is enabled.

Headless training omits rendering-only prediction IPC; recurrent minibatches
are gathered by complete environment sequences; attention masks stay on device;
unused attention scores are suppressed; optional compact legacy edge state
reduces storage; PPO metrics transfer once per update. Waffle GST histories
clear at episode boundaries and reset-observation penalties are not assigned
to terminal transitions. Shared changes also affect the original code path;
compare against the unchanged main branch for baseline reproduction.

Tests: python -m unittest discover -s tests

Still needed: measured perception and dynamics, wall geometry, physical ROS
adapter and independent watchdog/collision monitor, trajectory replay parity, compiled neural training and DDP. Do not
run the TurtleBot2i ROS adapter on the Waffle Pi. No speed or real-world
performance is claimed before benchmarks and supervised validation.

## Experimental GPU trajectory replay

Generate a pool using the existing native RVO2 solver on independent CPU
workers. Each trajectory retains its matching robot start/goal.

```sh
python train.py --waffle-pi --waffle-build-pool pools/train.npz --waffle-pool-count 1024 --num-processes 8
python train.py --waffle-pi --waffle-replay pools/train.npz --resume --fast-training --load-path trained_models/GST_predictor_rand/checkpoints/41665.pt --output_dir trained_models/waffle_replay --num-processes 256 --num-mini-batch 16 --num-env-steps 500000
```

The replay backend keeps the pool, dynamics, observations, GST and PPO
rollouts on the training device. No live ORCA solve or worker IPC occurs
during replay. Optional --waffle-compile-dynamics needs a separately validated
PyTorch 2.x environment. Test on small batches first; benchmark end-to-end
steps/sec and time to a held-out target, not just robot dynamics throughput.
Pool construction is CPU-parallel and memory scales with scenario count.
Use different seeds for held-out evaluation; evaluate against live ORCA too.

**Important:** replay is an approximate training backend, not exact original
ORCA equivalence: upstream human respawns and goal samples depend on robot
position, which is stationary during precomputation. It also introduces
square wall collisions and post-transition reward/termination. It does not
simulate sensor occlusion, tracking errors, actual floor geometry or
swept collision checks. No real hardware use is authorized by this model.
Keep the live Waffle CPU path as the reference. Disable
--use-proper-time-limits for replay. GST remains unchanged.

## Experimental ROS 2 shadow integration

These nodes are an initial implementation, not validated on hardware.
ROS 2 is assumed only for this adapter; a ROS 1 robot needs a separate
adapter. Use the actual Waffle Pi bringup, TF tree and detector package.
On a host with ROS 2 and compatible PyTorch/GST dependencies:

```sh
python -m waffle_pi.ros2_policy --gst-dir gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj
```

Default publishes **only** /crowdnav/cmd_vel_shadow. Input /odom,
timestamped detector PoseArray /dr_spaam_detections, and PoseStamped
/crowdnav/goal in odom coordinates are required. The policy applies
timestamped TF2 for detections; it does not provide AMCL, DR-SPAAM,
map filtering or calibration. Greedy fixed-slot tracking is provisional;
evaluate identity switches, loss of detection and temporal spacing.
Use synchronized clocks. The model must finish inference inside 250 ms.

On the Pi, start the gate in dry-run mode:
```sh
python -m waffle_pi.ros2_safety_gate
```
It outputs to a diagnostic topic, not /cmd_vel. The gate rejects stale
commands/scans, invalid or close ranges, reverse commands and software
e-stop; a true Bool on /crowdnav/estop latches it until restart. Its
stopping model is conservative but not certified; the stock lidar has
blind spots. No hardware testing was performed. Verify measured stopping
distance and actual Twist/TwistStamped interface, manual emergency stop,
exclusive command publisher, network/driver failure behavior and safety
of controlled empty-room trials before arming. The explicit
--enable-requests and --arm options are NOT a deployment recommendation
until those gates are passed. The original TurtleBot2i adapter stays
unchanged; this is not a full ROS integration or a person-safe system.
