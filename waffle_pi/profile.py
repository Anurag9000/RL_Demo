"""Provisional Waffle Pi simulation configuration."""
from .dynamics import DriveLimits
import json
import math


def configure(config, args):
    if args.env_name != 'CrowdSimPredRealGST-v0' or not args.sort_humans:
        raise ValueError("Waffle profile requires GST and sorted observations")
    config.waffle_pi = True
    if args.waffle_native_action:
        from .native import COMMISSION_V,COMMISSION_W,safety_parameters
        if not (args.waffle_live_orca or args.waffle_replay):
            raise ValueError("Native action training requires a tensor simulator")
        args.waffle_speed=min(args.waffle_speed,COMMISSION_V)
        args.waffle_angular_speed=min(args.waffle_angular_speed,COMMISSION_W)
        config.waffle_safety=safety_parameters(args.waffle_safety_calibration,args.waffle_radius)
    elif args.waffle_init_cartesian or args.waffle_safety_calibration:
        raise ValueError("Native transfer and simulated gate require native action mode")
    if args.waffle_scene_split!='train' and not args.waffle_eval_episodes:
        raise ValueError('Validation/test scenarios may not be used for training')
    if args.waffle_require_measured and args.waffle_eval_episodes and args.waffle_scene_split=='train':
        raise ValueError('Measured evaluation requires a held-out scene split')
    if args.waffle_detector_weights and not args.waffle_fidelity_profile:
        raise ValueError('Neural perception requires fidelity mode')
    if args.waffle_require_measured and not args.waffle_fidelity_profile:
        raise ValueError('Measured fidelity profile required')
    if args.waffle_fidelity_profile:
        from .fidelity import read_profile
        if not (args.waffle_native_action and args.waffle_live_orca):
            raise ValueError('Fidelity mode requires native live ORCA')
        if args.waffle_calibration or args.waffle_actuation_delay or args.waffle_compile_dynamics:
            raise ValueError('Fidelity profile owns actuator timing; do not combine with legacy calibration, delay or dynamics compilation')
        if args.waffle_require_measured and not args.waffle_safety_calibration:
            raise ValueError('Separate measured safety record required')
        config.fidelity,args.waffle_fidelity_sha=read_profile(args.waffle_fidelity_profile,args.waffle_require_measured)
        if args.waffle_require_measured and not config.fidelity.get('scene'):
            raise ValueError('Measured mode requires explicit surveyed scenarios')
        if args.waffle_require_measured and (not config.fidelity.get('detector_bins') or config.fidelity['detector_bins'][-1]['range_m'][1]<args.waffle_sensor_range):
            raise ValueError('Measured detector bins must cover policy range')
        if args.waffle_sensor_range>config.fidelity['laser']['range_max']:
            raise ValueError('Policy sensor range exceeds physical laser range')
        if args.waffle_detector_weights:
            import hashlib
            from pathlib import Path
            settings=config.fidelity.get('detector')
            if not settings:
                raise ValueError('Neural perception requires detector settings in fidelity profile')
            args.waffle_detector_sha=hashlib.sha256(Path(args.waffle_detector_weights).read_bytes()).hexdigest()
            if settings.get('weights_sha256')!=args.waffle_detector_sha:
                raise ValueError('Detector checkpoint hash mismatch')
            if args.waffle_noise or args.waffle_dropout or args.waffle_detection_delay:
                raise ValueError('Neural detector replaces statistical detector noise/dropout/delay')
            if args.num_processes>16:
                raise ValueError('Neural scan backend initially supports at most 16 independent detector streams')
        if config.fidelity.get('reward'):
            from .quality import validate_reward
            validate_reward(config.fidelity['reward'])
        laser=config.fidelity['laser']
        if math.hypot(laser['offset_x'],laser['offset_y'])>config.waffle_safety['sensor_offset']+1e-6:
            raise ValueError('Safety sensor offset must cover measured extrinsics')
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
        max_angular=args.waffle_angular_speed,
        max_wheel=0.26/0.033 if args.waffle_native_action else args.waffle_speed/0.033, motor_tau=args.waffle_motor_tau,
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
