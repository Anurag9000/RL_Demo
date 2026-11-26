"""Reference Gazebo world from the installed official ROBOTIS Jazzy SDF.

Static cylinders are sensor calibration targets, not simulated moving people.
No physical ROS bridge is started. This is a simulator qualification fixture,
not an RL environment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


MODEL='turtlebot3_waffle_pi'


def child(parent,tag,value=None,**attrs):
    node=ET.SubElement(parent,tag,attrs)
    if value is not None:
        node.text=str(value)
    return node


def solid(world,name,pose,kind,size):
    model=child(world,'model',name=name)
    child(model,'static','true')
    child(model,'pose',' '.join(map(str,pose)))
    link=child(model,'link',name='body')
    for tag in ('collision','visual'):
        shape=child(child(child(link,tag,name=tag),'geometry'),kind)
        if kind=='box':
            child(shape,'size',' '.join(map(str,size)))
        else:
            child(shape,'radius',size[0])
            child(shape,'length',size[1])


def build(source,arena=2.5,dt=.001,rectangles=(),targets=()):
    source=Path(source).resolve()
    if not math.isfinite(arena) or arena<1 or not .0005<=dt<=.01:
        raise ValueError('Invalid arena or physics timestep')
    model=ET.parse(source).getroot().find('model')
    if model is None or model.get('name')!=MODEL:
        raise ValueError('Expected official Waffle Pi model')
    # Resolve installed mesh resources, so copying the world elsewhere does
    # not accidentally depend on a worker's current directory.
    for uri in model.findall('.//mesh/uri'):
        if uri.text.startswith('model://'):
            mesh=source.parent.parent/uri.text[len('model://'):]
            if not mesh.is_file():
                raise FileNotFoundError(mesh)
            uri.text=mesh.resolve().as_uri()
    # Keep the original wheel controller and sensors. Isolate topics;
    # absolutely no base-link velocity override.
    for tag in ('topic','odom_topic','tf_topic'):
        for node in model.findall('.//'+tag):
            node.text='/crowdnav_sim/'+node.text.lstrip('/')
    root=ET.Element('sdf',version='1.9')
    world=child(root,'world',name='crowdnav')
    physics=child(world,'physics',name='reference',type='ignored')
    child(physics,'max_step_size',dt)
    child(physics,'real_time_factor',0)
    child(physics,'real_time_update_rate',0)
    for filename,name in [('physics','Physics'),('user-commands','UserCommands'),
                          ('scene-broadcaster','SceneBroadcaster'),('sensors','Sensors'),('imu','Imu')]:
        plugin=child(world,'plugin',filename='gz-sim-'+filename+'-system',name='gz::sim::systems::'+name)
        if name=='Sensors':
            child(plugin,'render_engine','ogre2')
    solid(world,'floor',[0,0,-.05,0,0,0],'box',[2*arena+2,2*arena+2,.1])
    for i,(x,y,sx,sy) in enumerate([(arena+.05,0,.1,2*arena+.2),(-arena-.05,0,.1,2*arena+.2),
                                      (0,arena+.05,2*arena,.1),(0,-arena-.05,2*arena,.1)]):
        solid(world,'wall_'+str(i),[x,y,.75,0,0,0],'box',[sx,sy,1.5])
    for i,(x0,y0,x1,y1) in enumerate(rectangles):
        if not all(math.isfinite(x) and abs(x)<=arena for x in (x0,y0,x1,y1)) or x1<=x0 or y1<=y0:
            raise ValueError('Invalid rectangle')
        solid(world,'obstacle_'+str(i),[(x0+x1)/2,(y0+y1)/2,.75,0,0,0],'box',[x1-x0,y1-y0,1.5])
    for i,(x,y,r) in enumerate(targets):
        if not all(math.isfinite(v) for v in (x,y,r)) or r<=0 or abs(x)+r>=arena or abs(y)+r>=arena:
            raise ValueError('Invalid calibration target')
        solid(world,'target_'+str(i),[x,y,.85,0,0,0],'cylinder',[r,1.7])
    world.append(model)
    return root


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-sdf',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--arena',type=float,default=2.5)
    p.add_argument('--dt',type=float,default=.001)
    p.add_argument('--scene-json',help='Optional rectangles_m and static_targets_m arrays')
    a=p.parse_args()
    scene=json.loads(Path(a.scene_json).read_text()) if a.scene_json else {}
    root=build(a.model_sdf,a.arena,a.dt,scene.get('rectangles_m',[]),scene.get('static_targets_m',[]))
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:
        ET.ElementTree(root).write(handle,encoding='utf-8',xml_declaration=True)
    print(json.dumps(dict(world=str(path),world_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(a.model_sdf).read_bytes()).hexdigest(),
        calibrated=False,physics_dt=a.dt),indent=2))


if __name__=='__main__':
    main()
