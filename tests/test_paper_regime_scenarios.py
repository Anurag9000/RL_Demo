import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'generate_paper_regime_scenarios.py'
SPEC=importlib.util.spec_from_file_location('paper_regime_scenarios',SCRIPT)
GEN=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class PaperRegimeScenarioTests(unittest.TestCase):
    def test_randomized_and_nonrandomized_human_contracts(self):
        randomized=GEN.human_contract('randomized')
        nonrandomized=GEN.human_contract('nonrandomized')
        self.assertTrue(randomized['random_goal_changing'])
        self.assertEqual(randomized['random_goal_change_interval_s'],5.0)
        self.assertEqual(randomized['random_goal_change_chance'],.5)
        self.assertFalse(nonrandomized['random_goal_changing'])
        self.assertTrue(randomized['end_goal_changing'])
        self.assertTrue(nonrandomized['end_goal_changing'])
        self.assertEqual(nonrandomized['end_goal_change_chance'],1.0)

    def test_sampled_traits_match_regime_ranges(self):
        rng=__import__('numpy').random.default_rng(3)
        random_case=GEN.sample_case(rng,'randomized','train',0,5.,.25)
        fixed_case=GEN.sample_case(rng,'nonrandomized','train',0,5.,.25)
        self.assertEqual(len(random_case['humans']),4)
        for human in random_case['humans']:
            self.assertGreaterEqual(human['speed'],.5)
            self.assertLessEqual(human['speed'],1.5)
            self.assertGreaterEqual(human['radius'],.3)
            self.assertLessEqual(human['radius'],.5)
        self.assertEqual({h['speed'] for h in fixed_case['humans']},{1.0})
        self.assertEqual({h['radius'] for h in fixed_case['humans']},{.3})

    def test_banks_are_disjoint_and_have_complete_splits(self):
        base={'schema':'waffle_fidelity_v1','status':'illustrative',
              'scene':{'arena_half_extent_m':5.,'rectangles_m':[],'scenarios':[]}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base_path=root/'base.json'
            base_path.write_text(json.dumps(base))
            outputs=[]
            for regime in ('randomized','nonrandomized'):
                result=GEN.build_profile(base_path,root/f'{regime}.json',regime,5)
                outputs.append(json.loads(Path(result['path']).read_text()))
            sets=[]
            for profile in outputs:
                scenarios=profile['scene']['scenarios']
                self.assertEqual(len(scenarios),15)
                by_split={split:[s for s in scenarios if s['split']==split]
                          for split in ('train','validation','test')}
                self.assertTrue(all(len(cases)==5 for cases in by_split.values()))
                sigs={GEN.signature(s) for s in scenarios}
                self.assertEqual(len(sigs),15)
                sets.append(sigs)
            self.assertTrue(sets[0].isdisjoint(sets[1]))


if __name__=='__main__':
    unittest.main()
