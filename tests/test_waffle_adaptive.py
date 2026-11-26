import random
import unittest
from types import SimpleNamespace
from scripts.train_adaptive import next_population,curriculum,target


class Adaptive(unittest.TestCase):
    def setUp(self):
        self.a=SimpleNamespace(mode='hybrid',eta=2,min_population=2,elite_fraction=.5,
          lr_min=1e-6,lr_max=3e-4,max_collision=.05,max_gate=.3,target_success=.85,
          curriculum=True,humans=4,noise=.03,dropout=.1,randomization=.2,
          occlusion=True,reactive=True,detection_delay=1,pose_drift=.002)
    def row(self,i,s,c):
        return dict(id=i,seed=i,status='ok',stage=1,state=f'state{i}',streak=1,
          own_transitions=20,hyperparameters=dict(lr=.0001,entropy_coef=.01,clip_param=.2),
          evaluation=dict(success_rate=s,collision_rate=c,gate_intervention_fraction=.1,mean_reward=1.))
    def test_curriculum(self):
        stages=curriculum(self.a)
        self.assertEqual([x['humans'] for x in stages],[1,2,4])
        self.assertFalse(stages[0]['occlusion'])
        self.assertEqual(stages[-1]['noise'],self.a.noise)
        self.assertTrue(stages[-1]['reactive'])
    def test_halving_and_exploit(self):
        rows=[self.row(0,.9,.0),self.row(1,.8,.0),self.row(2,1.,.2),self.row(3,.3,.0)]
        pop,pruned=next_population(rows,self.a,random.Random(3))
        self.assertEqual([x['id'] for x in pop],[0,1])
        self.assertEqual(set(pruned),{2,3})
        self.assertEqual(pop[1]['state'],'state0')
        self.assertTrue(pop[1]['retune'])
        self.assertEqual(pop[1]['streak'],0)
        self.assertTrue(target(rows[0],self.a))
        self.assertFalse(target(rows[2],self.a))
    def test_pure_modes(self):
        rows=[self.row(i,.9-i*.1,0.) for i in range(4)]
        self.a.mode='pbt'
        pop,pruned=next_population(rows,self.a,random.Random(3))
        self.assertEqual(len(pop),4)
        self.assertFalse(pruned)
        self.a.mode='halving'
        pop,pruned=next_population(rows,self.a,random.Random(3))
        self.assertEqual(len(pop),2)
        self.assertTrue(all(not p['retune'] for p in pop))
