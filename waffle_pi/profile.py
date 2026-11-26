"""Provisional Waffle Pi simulation configuration."""
from .dynamics import DriveLimits
import json
import math


def configure(config, args):
    if args.env_name != 'CrowdSimPredRealGST-v0' or not args.sort_humans:
        raise ValueError("Waffle profile requires GST and sorted observations")
    config.waffle_pi = True
    if args.waffle_calibration:
        with open(args.waffle_calibration) as handle:
            calibration=json.load(handle)
        if calibration.get('schema')!='waffle_motor_fit_v1':
            raise ValueError("Expected offline motor fit JSON")
        args.waffle_motor_tau=float(calibration['motor_tau'])
        args.waffle_wheel_accel=float(calibration['wheel_accel'])
        delay=float(calibration['command_delay_s'])
        if not math.isfinite(delay) or not 0<=delay<=.5:
            raise ValueError("Invalid fitted command delay")
        args.waffle_actuation_delay=int(round(delay/.25))
    if not 0<=args.waffle_actuation_delay<=8 or not 0<=args.waffle_dynamics_randomization<=.5:
        raise ValueError("Invalid command delay or dynamics randomization")
    if (args.waffle_actuation_delay or args.waffle_dynamics_randomization) and not (args.waffle_live_orca or args.waffle_replay):
        raise ValueError("Actuation delay and dynamics randomization require tensor simulator")
    config.waffle_limits = DriveLimits(max_linear=args.waffle_speed,
        max_wheel=args.waffle_speed/0.033, motor_tau=args.waffle_motor_tau,
        wheel_accel=args.waffle_wheel_accel)
    config.action_space.kinematics = 'unicycle'
    config.robot.v_pref = args.waffle_speed
    config.robot.radius = args.waffle_radius
    config.robot.sensor_range = args.waffle_sensor_range
    config.robot.visible = False
    config.sim.arena_size = args.waffle_arena
    config.sim.circle_radius = args.waffle_arena
    config.sim.human_num = args.waffle_humans
    config.sim.human_num_range = 0
    config.env.time_limit = args.waffle_timeout
    config.sim2real.use_dummy_detect = False
    config.waffle_min_goal = 1.5
    config.waffle_max_goal = min(4., 2*(args.waffle_arena-args.waffle_radius))
    if (not 0 < args.waffle_radius or not 1 <= args.waffle_humans <= 20
            or not 0 < args.waffle_sensor_range <= 5
            or config.waffle_max_goal <= 1.5
            or args.waffle_timeout <= config.waffle_max_goal/args.waffle_speed):
        raise ValueError("Invalid Waffle profile or infeasible episode")
