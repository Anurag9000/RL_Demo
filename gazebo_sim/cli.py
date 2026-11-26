"""Shared CLI contract for Gazebo experiment runners."""
def add_flags(p):
    p.add_argument('--gazebo-model')
    p.add_argument('--gazebo-worker',default='build/gazebo/waffle_gazebo_worker')
    p.add_argument('--gazebo-logs',default='gazebo_runs/workers')
    p.add_argument('--gazebo-timeout',type=float,default=120.)
    p.add_argument('--gazebo-detector-device',default='cpu')


def backend_flags(a):
    if a.backend!='gazebo':
        return ['--waffle-live-orca']
    if not a.gazebo_model or not a.fidelity_profile or not a.detector_weights:
        raise ValueError('Gazebo requires --gazebo-model --fidelity-profile --detector-weights')
    return ['--waffle-gazebo','--waffle-gazebo-model',a.gazebo_model,
            '--waffle-gazebo-worker',a.gazebo_worker,'--waffle-gazebo-logs',a.gazebo_logs,
            '--waffle-gazebo-timeout',str(a.gazebo_timeout),
            '--waffle-gazebo-detector-device',a.gazebo_detector_device]
