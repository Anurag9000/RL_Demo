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
adapter and independent watchdog/collision monitor, fully GPU-resident
simulator, trajectory replay parity, compiled neural training and DDP. Do not
run the TurtleBot2i ROS adapter on the Waffle Pi. No speed or real-world
performance is claimed before benchmarks and supervised validation.
