import unittest
import tempfile
from pathlib import Path
import numpy as np
from waffle_pi.paper_metrics import intrusion_metrics, aggregate_intrusion_metrics
from waffle_pi.offline_metrics import score_trace_directory


class PaperMetrics(unittest.TestCase):
    def test_itr_uses_future_truth_sd_uses_current_time(self):
        robot = np.array([[0., 0.], [0., 0.], [0., 0.]])
        # At t=0 future truth intrudes, but current closest center distance is 3m.
        # At t=1 no future truth intrudes. Initial truth is deliberately distinct.
        humans = np.array([[[3., 0.]], [[.5, 0.]], [[3., 0.]], [[3., 0.]], [[3., 0.]], [[3., 0.]], [[3., 0.]]])
        result = intrusion_metrics(robot, humans, [.3], .25, future_steps=5)
        self.assertEqual(result['paper_intrusion_steps'], 1)
        self.assertEqual(result['paper_episode_steps'], 2)
        self.assertEqual(result['paper_itr'], .5)
        self.assertAlmostEqual(result['paper_sd_m'], 3.)
        self.assertAlmostEqual(result['paper_sd_sum_m'], 3.)
        self.assertEqual(result['paper_sd_count'], 1)

    def test_current_closest_human_is_selected(self):
        robot = np.zeros((2, 2))
        humans = np.full((6, 2, 2), 4.)
        humans[0] = [[1., 0.], [.2, 0.]]
        # Exactly radius sum at future t=1 is not intrusion; second human overlaps.
        humans[1] = [[.55, 0.], [.1, 0.]]
        result = intrusion_metrics(robot, humans, [.3, .3], .25, 5)
        self.assertEqual(result['paper_intrusion_steps'], 1)
        self.assertAlmostEqual(result['paper_sd_m'], .2)

    def test_contact_boundary_is_not_an_intrusion(self):
        robot = np.zeros((2, 2))
        humans = np.full((7, 1, 2), [.55, 0.])
        result = intrusion_metrics(robot, humans, [.3], .25, 5)
        self.assertEqual(result['paper_intrusion_steps'], 0)
        self.assertEqual(result['paper_itr'], 0.)
        self.assertIsNone(result['paper_sd_m'])

    def test_requires_complete_ground_truth_horizon(self):
        with self.assertRaises(ValueError):
            intrusion_metrics(np.zeros((2, 2)), np.zeros((5, 1, 2)), [.3], .25, 5)

    def test_no_intrusions_has_undefined_sd(self):
        result = intrusion_metrics(np.zeros((2, 2)), np.full((7, 1, 2), 4.), [.3], .25, 5)
        self.assertEqual(result['paper_itr'], 0.)
        self.assertIsNone(result['paper_sd_m'])
        self.assertEqual(result['paper_sd_count'], 0)

    def test_offline_trace_scorer_recomputes_and_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            robot=np.zeros((2,2),dtype=np.float32)
            base=dict(robot_positions=robot,human_radii=np.asarray([.3]),human_ids=np.asarray(['person_0']),
                robot_times=np.asarray([2.,2.25]),human_times=np.arange(6,dtype=np.float64)*.25+2.,
                robot_radius=np.asarray(.25),future_steps=np.asarray(5),episode_steps=np.asarray(1),
                scenario_index=np.asarray(0),scenario_id=np.asarray('case'),split=np.asarray('validation'),
                seed=np.asarray(7),lane=np.asarray(0),episode_number=np.asarray(1),profile_sha256=np.asarray('profile'),
                checkpoint_sha256=np.asarray('checkpoint'),detector_sha256=np.asarray('detector'),
                model_sdf_sha256=np.asarray('model'),worker_sha256=np.asarray('worker'),
                online_sd_m=np.asarray(0.),online_sd_valid=np.asarray(False))
            humans_clear=np.full((6,1,2),4.,dtype=np.float32)
            np.savez_compressed(root/'clear.npz',human_positions=humans_clear,online_itr=np.asarray(0.),**base)
            humans_intrude=humans_clear.copy(); humans_intrude[0,0]=[1.,0.]; humans_intrude[1,0]=[.1,0.]
            base_intrude=dict(base,online_itr=np.asarray(1.),online_sd_m=np.asarray(1.),
                              online_sd_valid=np.asarray(True),scenario_id=np.asarray('intrusion'))
            base_intrude['human_positions']=humans_intrude
            np.savez_compressed(root/'intrusion.npz',**base_intrude)
            result=score_trace_directory(root)
            self.assertEqual(result['trace_count'],2)
            self.assertAlmostEqual(result['mean_episode_itr'],.5)
            self.assertEqual(result['intrusion_steps'],1)
            self.assertAlmostEqual(result['pooled_sd_m'],1.)
            self.assertTrue(result['all_online_parity'])
            self.assertEqual(result['episodes'][0]['checkpoint_sha256'],'checkpoint')
            self.assertEqual(result['episodes'][0]['human_ids'],['person_0'])
            self.assertEqual(result['episodes'][0]['human_time_end'],3.25)

    def test_pooled_metrics_weight_intrusion_events_and_timesteps(self):
        pooled = aggregate_intrusion_metrics([
            {'paper_itr': .5, 'paper_intrusion_steps': 1, 'paper_episode_steps': 2, 'paper_sd_sum_m': 2.},
            {'paper_itr': .3, 'paper_intrusion_steps': 3, 'paper_episode_steps': 10, 'paper_sd_sum_m': 12.},
        ])
        self.assertEqual(pooled['paper_intrusion_steps'], 4)
        self.assertEqual(pooled['paper_episode_steps'], 12)
        self.assertAlmostEqual(pooled['paper_itr'], (.5+.3)/2)
        self.assertAlmostEqual(pooled['paper_sd_m'], 14/4)


if __name__ == '__main__':
    unittest.main()
