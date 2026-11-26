import tempfile
import unittest
from pathlib import Path
from waffle_pi.gazebo_world import build


class World(unittest.TestCase):
    def test_original_model_and_isolated_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.sdf'
            path.write_text('<sdf version="1.8"><model name="turtlebot3_waffle_pi"><plugin filename="gz-sim-diff-drive-system"><wheel_radius>0.033</wheel_radius><topic>cmd_vel</topic><tf_topic>/tf</tf_topic></plugin></model></sdf>')
            root=build(path,rectangles=[[-1.,-1.,-.5,-.5]],targets=[[1.,1.,.2]])
            model=root.find("world/model[@name='turtlebot3_waffle_pi']")
            self.assertEqual(model.findtext('plugin/wheel_radius'),'0.033')
            self.assertEqual(model.findtext('plugin/topic'),'/crowdnav_sim/cmd_vel')
            self.assertEqual(model.findtext('plugin/tf_topic'),'/crowdnav_sim/tf')
            self.assertIsNotNone(root.find("world/model[@name='target_0']"))
            with self.assertRaises(ValueError):
                build(path,rectangles=[[1.,1.,0.,0.]])
