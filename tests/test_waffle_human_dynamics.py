import unittest
from types import SimpleNamespace
import numpy as np
from waffle_pi.human_dynamics import update_human_goals,validate_human_dynamics
from gazebo_sim.vector_env import PerWorld


class HumanDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.positions=np.array([[-1.,0.],[1.,0.]])
        self.goals=np.array([[-2.,0.],[2.,0.]])
        self.radii=np.array([.3,.3])
        self.robot=np.array([0.,-2.])
        self.robot_goal=np.array([0.,2.])

    def update(self,tick,config,seed=7,positions=None,goals=None):
        return update_human_goals(self.positions if positions is None else positions,
            self.goals if goals is None else goals,self.radii,self.robot,self.robot_goal,
            .25,5.,tick,.25,config,np.random.default_rng(seed))

    def test_nonrandomized_keeps_goal_until_arrival_then_changes(self):
        cfg=dict(regime='nonrandomized',random_goal_changing=False,
                 end_goal_changing=True,end_goal_change_chance=1.)
        goals,changed,failed=self.update(20,cfg)
        self.assertEqual(changed,0)
        self.assertEqual(failed,0)
        arrived=self.positions.copy(); arrived[0]=self.goals[0]
        goals,changed,failed=self.update(21,cfg,positions=arrived)
        self.assertEqual(changed,1)
        self.assertEqual(failed,0)
        self.assertFalse(np.array_equal(goals[0],self.goals[0]))

    def test_randomized_mid_episode_changes_follow_interval_and_are_seeded(self):
        cfg=dict(regime='randomized',random_goal_changing=True,
                 random_goal_change_interval_s=1.,random_goal_change_chance=1.,
                 end_goal_changing=False)
        early,changed,_=self.update(2,cfg)
        self.assertEqual(changed,0)
        first=self.update(3,cfg,seed=11)
        second=self.update(3,cfg,seed=11)
        self.assertEqual(first[1],2)
        np.testing.assert_array_equal(first[0],second[0])
        self.assertFalse(np.array_equal(first[0],self.goals))

    def test_goal_sampling_respects_arena_and_robot_clearance(self):
        cfg=dict(regime='nonrandomized',random_goal_changing=False,
                 end_goal_changing=True,end_goal_change_chance=1.,goal_clearance_m=.2)
        positions=self.positions.copy(); positions[0]=self.goals[0]
        goals,changed,failed=self.update(1,cfg,positions=positions)
        self.assertEqual(changed,1)
        self.assertEqual(failed,0)
        self.assertLessEqual(np.abs(goals[0]).max(),5.-.3-.2)
        self.assertGreaterEqual(np.linalg.norm(goals[0]-self.robot),.3+.2+.25)
        self.assertGreaterEqual(np.linalg.norm(goals[0]-positions[1]),.3+.2+.3)

    def test_invalid_interval_is_rejected(self):
        cfg=dict(regime='randomized',random_goal_changing=True,
                 random_goal_change_interval_s=.3,random_goal_change_chance=.5)
        with self.assertRaisesRegex(ValueError,'whole number'):
            self.update(4,cfg)

    def test_five_second_change_occurs_on_twentieth_policy_boundary(self):
        cfg=dict(regime='randomized',random_goal_changing=True,
                 random_goal_change_interval_s=5.,random_goal_change_chance=1.,
                 end_goal_changing=False)
        before=self.update(18,cfg)
        at_boundary=self.update(19,cfg)
        self.assertEqual(before[1],0)
        self.assertEqual(at_boundary[1],2)

    def test_per_world_applies_goal_change_at_five_second_boundary(self):
        world=object.__new__(PerWorld)
        world.human_dynamics=dict(regime='randomized',random_goal_changing=True,
            random_goal_change_interval_s=5.,random_goal_change_chance=1.,
            end_goal_changing=False)
        world.current_human_goals=self.goals.copy()
        world.radius=self.radii.copy()
        world.goal=self.robot_goal.copy()
        world.state={'truth':np.array([self.robot[0],self.robot[1],0.])}
        world.cfg=SimpleNamespace(robot=SimpleNamespace(radius=.25),
                                  env=SimpleNamespace(time_step=.25))
        world.profile={'scene':{'arena_half_extent_m':5.,'rectangles_m':[]}}
        world.n=2
        world.policy_tick=19
        world.rng=np.random.default_rng(7)
        world.human_goal_change_count=0
        world.human_goal_sample_failures=0
        result=world.next_human_goals(self.positions)
        self.assertEqual(world.policy_tick,20)
        self.assertEqual(world.human_goal_change_count,2)
        self.assertEqual(world.human_goal_sample_failures,0)
        self.assertFalse(np.array_equal(result,self.goals))

    def test_invalid_nonfinite_timing_is_rejected(self):
        cfg=dict(regime='randomized',random_goal_changing=True,
                 random_goal_change_interval_s=5.,random_goal_change_chance=.5)
        with self.assertRaisesRegex(ValueError,'robot, arena or timestep'):
            update_human_goals(self.positions,self.goals,self.radii,self.robot,
                self.robot_goal,.25,float('nan'),0,.25,cfg,np.random.default_rng(1))

    def test_regime_schema_validation(self):
        with self.assertRaisesRegex(ValueError,'randomized or nonrandomized'):
            validate_human_dynamics(dict(regime='mixed'))


if __name__=='__main__':
    unittest.main()
