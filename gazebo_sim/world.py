
"""Training scenes built from the installed official Jazzy Waffle Pi SDF."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from waffle_pi.gazebo_world import build,child,MODEL


def replace(parent,tag,value):
    node=parent.find(tag)
    if node is None:
        node=child(parent,tag)
    node.text=str(value)


def validate_gazebo_profile(profile,measured=False):
    g=profile.get('gazebo')
    if not isinstance(g,dict):
        raise ValueError('Fidelity profile needs a gazebo section')
    dt=float(g['physics_dt'])
    control=float(g['gate_dt'])
    if not .0005<=dt<=.005 or not .005<=control<=.05 or abs(control/dt-round(control/dt))>1e-7 or abs(.25/control-round(.25/control))>1e-7:
        raise ValueError('Physics/gate intervals must divide the policy interval exactly')
    for key,low,high in [('velocity_kp',.001,20.),('velocity_ki',0.,50.),('torque_limit_nm',.01,3.)]:
        pair=profile['motor'][key]
        if len(pair)!=2 or any(not math.isfinite(x) or not low<=x<=high for x in pair):
            raise ValueError('Invalid bounded actuator parameter '+key)
    if not .01<=g['wheel_friction']<=10 or not 0<=g['wheel_slip']<=1:
        raise ValueError('Invalid calibrated wheel contact parameters')
    if not .025<=g['leg_radius_m']<=.15 or not .05<=g['leg_separation_m']<=.4:
        raise ValueError('Invalid pedestrian leg proxy')
    if measured and (not g.get('physics_calibration_record') or not g.get('model_sha256') or not g.get('worker_sha256')):
        raise ValueError('Measured Gazebo requires physics evidence and model/worker hashes')
    if not profile.get('scene'):
        raise ValueError('Gazebo requires an explicit scenario bank')
    return g


def source_digest(source):
    # Include installed mesh contents as well as original SDF, not just XML.
    source=Path(source).resolve()
    digest=hashlib.sha256(source.read_bytes())
    model=ET.parse(source).getroot().find('model')
    if model is None or model.get('name')!=MODEL:
        raise ValueError('Only official TurtleBot3 Waffle Pi is accepted')
    if any(model.find("joint[@name='"+j+"']") is None for j in ('wheel_left_joint','wheel_right_joint','base_joint')):
        raise ValueError('Missing official wheel or base joints')
    if any(model.find(".//sensor[@type='"+s+"']") is None for s in ('gpu_lidar','imu','camera')):
        raise ValueError('Missing official Waffle Pi sensors')
    for node in model.findall('.//mesh/uri'):
        if not node.text.startswith('model://'):
            raise ValueError('Expected installed model:// mesh resources')
        path=source.parent.parent/node.text[len('model://'):]
        digest.update(node.text.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def person(world,i,h,g):
    m=child(world,'model',name='person_'+str(i))
    x,y=h['start']
    child(m,'pose',f'{x} {y} 0 0 0 0')
    link=child(m,'link',name='body')
    child(link,'gravity','false')
    inert=child(link,'inertial')
    child(inert,'mass',65.)
    child(inert,'pose','0 0 .85 0 0 0')
    inertia=child(inert,'inertia')
    for tag,value in [('ixx',15.),('iyy',15.),('izz',2.),('ixy',0.),('ixz',0.),('iyz',0.)]:
        child(inertia,tag,value)
    # Two leg surfaces are visible at LDS height; torso is higher up.
    # Fixed leg geometry is a proxy, not an articulated walking person.
    for name,z,dy,r,length in [('left_leg',.32,g['leg_separation_m']/2,g['leg_radius_m'],.64),
                               ('right_leg',.32,-g['leg_separation_m']/2,g['leg_radius_m'],.64),
                               ('torso',1.05,0.,h['radius'],.8),('head',1.57,0.,.10,.24)]:
        for kind in ('collision','visual'):
            node=child(link,kind,name=name+'_'+kind)
            child(node,'pose',f'0 {dy} {z} 0 0 0')
            cyl=child(child(node,'geometry'),'cylinder')
            child(cyl,'radius',r)
            child(cyl,'length',length)
    return m


def training_world(source,profile,scenario):
    g=validate_gazebo_profile(profile)
    scene=profile['scene']
    root=build(source,scene['arena_half_extent_m'],g['physics_dt'],scene['rectangles_m'])
    world=root.find('world')
    robot=world.find("model[@name='"+MODEL+"']")
    start=scenario['robot_start']
    replace(robot,'pose',f'{start[0]} {start[1]} 0 0 0 {start[2]}')
    # Remove only the stock DiffDrive command system, avoiding two competing
    # wheel writers. Keep the official geometry, joints, sensor and IMU model.
    drives=[p for p in robot.findall('plugin') if p.get('name')=='gz::sim::systems::DiffDrive']
    if len(drives)!=1:
        raise ValueError('Expected exactly one official DiffDrive plugin')
    robot.remove(drives[0])
    for name in ('wheel_left_link','wheel_right_link'):
        link=robot.find("link[@name='"+name+"']")
        ode=link.find('collision/surface/friction/ode')
        for k,v in [('mu',g['wheel_friction']),('mu2',g['wheel_friction']),('slip1',g['wheel_slip']),('slip2',g['wheel_slip'])]:
            replace(ode,k,v)
    laser=profile['laser']
    sensors=robot.findall(".//sensor[@type='gpu_lidar']")
    if len(sensors)!=1:
        raise ValueError('Expected one official LDS')
    sensor=sensors[0]
    pose=list(map(float,sensor.findtext('pose').split()))
    pose[0],pose[1],pose[5]=laser['offset_x'],laser['offset_y'],laser['yaw']
    if 'height_m' in laser:
        pose[2]=laser['height_m']
    replace(sensor,'pose',' '.join(map(str,pose)))
    replace(sensor,'update_rate',laser['hz'])
    replace(sensor,'topic','/crowdnav_sim/scan')
    ray=sensor.find('ray')
    if ray is None:
        ray=sensor.find('lidar')
    horizontal=ray.find('scan/horizontal')
    replace(horizontal,'samples',laser['rays'])
    replace(horizontal,'resolution',1)
    replace(horizontal,'min_angle',laser['angle_min'])
    replace(horizontal,'max_angle',laser['angle_min']+2*math.pi*(laser['rays']-1)/laser['rays'])
    for k,v in [('min',laser['range_min']),('max',laser['range_max'])]:
        replace(ray.find('range'),k,v)
    if 'range_resolution_m' in laser:
        replace(ray.find('range'),'resolution',laser['range_resolution_m'])
    replace(ray.find('noise'),'mean',laser.get('range_bias_m',0.))
    replace(ray.find('noise'),'stddev',laser['noise_std_m'])
    # Camera and IMU remain in the SDF. The navigation policy does not
    # consume camera pixels; no artificial camera observations are substituted.
    for i,h in enumerate(scenario['humans']):
        person(world,i,h,g)
    return root


def worker_config(profile,scenario):
    return dict(motor=profile['motor'],start=scenario['robot_start'],
                humans=len(scenario['humans']),dt=profile['gazebo']['physics_dt'],
                laser_hz=profile['laser']['hz'])


def write_case(directory,source,profile,scenario,seed=0):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    world=directory/'world.sdf'
    config=directory/'worker.json'
    ET.ElementTree(training_world(source,profile,scenario)).write(world,encoding='utf-8',xml_declaration=True)
    config.write_text(json.dumps(dict(worker_config(profile,scenario),seed=int(seed)),allow_nan=False))
    return world,config
