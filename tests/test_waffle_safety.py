import unittest
from waffle_pi.safety import safe_command


class TestSafety(unittest.TestCase):
    def command(self, ranges=(2.,), **kwargs):
        args=dict(fresh=True,armed=True,estop=False)
        args.update(kwargs)
        return safe_command(.26,1.82,ranges,.12,3.5,**args)

    def test_caps(self):
        v,w=self.command()
        self.assertLessEqual(v,.100001)
        self.assertLessEqual(w,.500001)
        self.assertGreater(v,0.)

    def test_fail_closed(self):
        for kwargs in ({'fresh':False},{'armed':False},{'estop':True}):
            self.assertEqual(self.command(**kwargs),(0.,0.))
        for ranges in ((.2,),(.01,),(float('nan'),),(float('inf'),),()):
            self.assertEqual(self.command(ranges),(0.,0.))
