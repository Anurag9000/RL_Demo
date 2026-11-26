import unittest
from scripts.train_portfolio import rank_candidates


class Portfolio(unittest.TestCase):
    def test_collision_filter_and_success(self):
        def row(i,s,c):
            return dict(index=i,status='ok',wall_s=1.,evaluation=dict(success_rate=s,
                collision_rate=c,gate_intervention_fraction=.1,mean_reward=1.))
        self.assertEqual([r['index'] for r in rank_candidates(
            [row(0,.99,.1),row(1,.7,0.),row(2,.8,.02)],.05)],[2,1])
