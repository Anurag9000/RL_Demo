import unittest
import numpy as np
from waffle_pi.paper_metrics import intrusion_metrics, aggregate_intrusion_metrics


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
