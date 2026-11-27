import os
import json
import mmap
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from gazebo_sim.hardware import parse_cpus,format_cpus,worker_command
from gazebo_sim.worker import Worker
from scripts.hardware_sweep import modes,summary


class Hardware(unittest.TestCase):
    def test_cpu_sets(self):
        self.assertEqual(parse_cpus('0-2,7,9-10'),{0,1,2,7,9,10})
        self.assertEqual(format_cpus({5,1,2}),'1,2,5')
        with self.assertRaises(ValueError):
            parse_cpus('7-2')
        with patch('gazebo_sim.hardware.numa_nodes',return_value={0:[0,1,2,3],1:[4,5,6,7]}),patch('gazebo_sim.hardware.os.sched_getaffinity',return_value=set(range(8))),patch('gazebo_sim.hardware.shutil.which',return_value='/usr/bin/taskset'):
            cmd=worker_command(['worker'],dict(WAFFLE_WORKER_AFFINITY='1',WAFFLE_WORKER_CORES='2',WAFFLE_RESERVE_CORES='0'),2)
            self.assertEqual(cmd,['taskset','-c','4,5','worker'])
    def test_podman_worker_command_mounts_only_world_and_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            world_dir=root/'episode'; world_dir.mkdir()
            model_dir=root/'models'; model_dir.mkdir()
            world=world_dir/'world.sdf'; world.write_text('<sdf/>')
            config=world_dir/'worker.json'; config.write_text('{}')
            env=dict(WAFFLE_GAZEBO_PODMAN_IMAGE='localhost/gz:mission',
                     GZ_SIM_RESOURCE_PATH=f'{world_dir}:{model_dir}',
                     GZ_PARTITION='mission-test',ROS_LOCALHOST_ONLY='1',OMP_NUM_THREADS='1')
            with patch('gazebo_sim.hardware.shutil.which',return_value='/usr/bin/podman'):
                cmd=worker_command(['unused',str(world),str(config)],env,0)
            self.assertEqual(cmd[0],'/usr/bin/podman')
            self.assertIn(f'{world_dir}:{world_dir}:rw',cmd)
            self.assertIn(f'{model_dir}:{model_dir}:ro',cmd)
            self.assertIn('GZ_PARTITION=mission-test',cmd)
            self.assertEqual(cmd[-2:], [str(world),str(config)])

    def test_modes_and_tradeoffs(self):
        info=dict(nodes={0:list(range(32)),1:list(range(32,64))},
                  gpus={0:dict(numa=0),1:dict(numa=1)},affinity=list(range(64)))
        plan=modes(info,[0,1],48,4)
        self.assertEqual(plan[0]['envs'],48)
        self.assertEqual(len(plan[0]['cpus'][0]),64)
        self.assertEqual(plan[3]['envs']*2,48)
        self.assertEqual(plan[3]['minibatches'],4)
        self.assertFalse(set(plan[3]['cpus'][0])&set(plan[3]['cpus'][1]))
        rows=[dict(mode=m,preset='eager',global_envs=48,status='ok',
            policy_transitions_per_sec=s,aggregate_transitions_per_sec=a,wall_transitions_per_sec=s)
            for m,s,a in [('single-all-cpus',100,100),('ddp',150,150),('independent',80,160)]]
        result=summary(rows)
        self.assertEqual(result['fastest_single_policy']['mode'],'ddp')
        self.assertEqual(result['matched_batch_tradeoffs'][0]['ddp_speedup'],1.5)
        self.assertEqual(result['matched_batch_tradeoffs'][0]['independent_search_throughput_ratio'],1.6)
    def test_scan_buffer_snapshot(self):
        worker=Worker.__new__(Worker)
        worker.stats={'scan_bytes':0}
        worker.scan_map=mmap.mmap(-1,256*4096*4)
        try:
            view=np.array([1.,np.inf,.25],dtype=np.float32)
            worker.scan_map.write(view.tobytes())
            msg={'scans':[{'range_offset':0,'range_count':3}]}
            worker._read_scans(msg)
            np.testing.assert_array_equal(msg['scans'][0]['ranges'],view)
            worker.scan_map.seek(0)
            worker.scan_map.write(np.zeros(3,np.float32).tobytes())
            np.testing.assert_array_equal(msg['scans'][0]['ranges'],view)
            with self.assertRaises(RuntimeError):
                worker._read_scans({'scans':[{'range_offset':1,'range_count':3}]})
            self.assertEqual(worker.stats['scan_bytes'],12)
        finally:
            worker.scan_map.close()
