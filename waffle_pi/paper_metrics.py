"""Ground-truth social metrics used by the reference navigation paper."""
import numpy as np


def intrusion_metrics(robot_positions, human_positions, human_radii, robot_radius, future_steps=5):
    """Measure future-truth intrusion over a complete episode trajectory.

    ``robot_positions`` and ``human_positions`` include the episode's initial
    state and every policy-step state. ``human_positions`` must additionally
    contain ``future_steps`` states beyond termination so every episode step
    has a complete lookahead. Distances in SD are center-to-center distances,
    matching the paper's stated social-distance metric.
    """
    robot = np.asarray(robot_positions, dtype=np.float64)
    humans = np.asarray(human_positions, dtype=np.float64)
    radii = np.asarray(human_radii, dtype=np.float64)
    if robot.ndim != 2 or robot.shape[1] != 2 or not np.isfinite(robot).all():
        raise ValueError("robot_positions must be finite [T+1, 2]")
    if humans.ndim != 3 or humans.shape[2] != 2 or not np.isfinite(humans).all():
        raise ValueError("human_positions must be finite [T+1+future_steps, N, 2]")
    if len(radii) != humans.shape[1] or not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError("human_radii must contain one positive radius per human")
    if not np.isfinite(robot_radius) or robot_radius <= 0 or type(future_steps) is not int or future_steps < 1:
        raise ValueError("robot_radius and future_steps must be positive")
    episode_steps = len(robot) - 1
    if episode_steps < 1 or len(humans) < len(robot) + future_steps - 1:
        raise ValueError("trajectory does not cover every episode step and future horizon")
    if humans.shape[1] == 0:
        raise ValueError("at least one human is required")

    intrusion_steps = 0
    sd_sum = 0.0
    contact_distance = radii + robot_radius
    for t in range(episode_steps):
        # ITR uses future ground-truth human positions against the current
        # robot center. SD is the closest center distance at that same
        # policy timestep, and is accumulated only for intrusion timesteps.
        future = humans[t + 1:t + 1 + future_steps]
        future_distances = np.linalg.norm(future - robot[t][None, None, :], axis=-1)
        is_intrusion = bool(np.any(future_distances < contact_distance[None, :]))
        if is_intrusion:
            current_distances = np.linalg.norm(humans[t] - robot[t][None, :], axis=-1)
            sd_sum += float(current_distances.min())
            intrusion_steps += 1
    return {
        "paper_intrusion_steps": intrusion_steps,
        "paper_episode_steps": episode_steps,
        "paper_itr": intrusion_steps / episode_steps,
        "paper_sd_sum_m": sd_sum,
        "paper_sd_count": intrusion_steps,
        "paper_sd_m": sd_sum / intrusion_steps if intrusion_steps else None,
    }


def aggregate_intrusion_metrics(episode_metrics):
    """Average per-episode ITR and pool SD over intrusion timesteps, as in the paper."""
    episodes = list(episode_metrics)
    steps = sum(int(item["paper_episode_steps"]) for item in episodes)
    intrusions = sum(int(item["paper_intrusion_steps"]) for item in episodes)
    sd_sum = sum(float(item["paper_sd_sum_m"]) for item in episodes)
    return {
        "paper_itr": float(np.mean([float(item["paper_itr"]) for item in episodes])) if episodes else None,
        "paper_sd_m": sd_sum / intrusions if intrusions else None,
        "paper_intrusion_steps": intrusions,
        "paper_episode_steps": steps,
    }
