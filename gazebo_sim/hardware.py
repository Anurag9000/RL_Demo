"""Linux affinity helpers, respecting the caller's existing cpuset."""
import os
from pathlib import Path
import shutil


def parse_cpus(spec):
    cpus=set()
    for item in spec.split(','):
        a,sep,b=item.strip().partition('-')
        lo,hi=int(a),int(b) if sep else int(a)
        if lo<0 or hi<lo or hi-lo>65536:
            raise ValueError('Invalid CPU set')
        cpus.update(range(lo,hi+1))
    if not cpus:
        raise ValueError('Empty CPU set')
    return cpus


def format_cpus(cpus):
    return ','.join(map(str,sorted(cpus)))


def numa_nodes(allowed=None):
    allowed=set(os.sched_getaffinity(0) if allowed is None else allowed)
    nodes={}
    for path in Path('/sys/devices/system/node').glob('node[0-9]*'):
        cpus=parse_cpus((path/'cpulist').read_text().strip())&allowed
        if cpus:
            nodes[int(path.name[4:])]=sorted(cpus)
    return nodes or {0:sorted(allowed)}


def bind_rank():
    spec=os.environ.get('WAFFLE_RANK_CPUSETS')
    if not spec:
        return
    rank=int(os.environ.get('LOCAL_RANK','0'))
    sets=spec.split(';')
    if rank>=len(sets):
        raise ValueError('No cpuset for local rank')
    cpus=parse_cpus(sets[rank])
    if not cpus<=os.sched_getaffinity(0):
        raise ValueError('Requested rank CPUs outside inherited allocation')
    os.sched_setaffinity(0,cpus)


def worker_command(cmd,env,index):
    image=env.get('WAFFLE_GAZEBO_PODMAN_IMAGE')
    if image:
        if len(cmd)!=3:
            raise ValueError('Podman Gazebo worker requires binary, world and config paths')
        if env.get('WAFFLE_SCAN_TRANSPORT','json')!='json':
            raise ValueError('Podman Gazebo worker currently requires JSON scan transport')
        podman=shutil.which('podman')
        if not podman:
            raise RuntimeError('WAFFLE_GAZEBO_PODMAN_IMAGE requires rootless podman')
        world,config=Path(cmd[1]).resolve(),Path(cmd[2]).resolve()
        if not world.is_file() or not config.is_file():
            raise FileNotFoundError('Podman Gazebo worker world/config is missing')
        world_dir=world.parent
        resources=[]
        for raw in env.get('GZ_SIM_RESOURCE_PATH','').split(os.pathsep):
            if raw:
                resource=Path(raw).resolve()
                if not resource.exists():
                    raise FileNotFoundError('Gazebo resource path is missing: '+str(resource))
                if resource!=world_dir and resource not in resources:
                    resources.append(resource)
        wrapped=[podman,'run','--rm','--interactive','--network','host',
                 '--volume',f'{world_dir}:{world_dir}:rw']
        wrapped.extend(arg for resource in resources
                       for arg in ('--volume',f'{resource}:{resource}:ro'))
        for key in ('GZ_PARTITION','GZ_SIM_RESOURCE_PATH','ROS_LOCALHOST_ONLY','OMP_NUM_THREADS'):
            if env.get(key) is not None:
                wrapped.extend(('--env',key+'='+env[key]))
        worker=env.get('WAFFLE_GAZEBO_CONTAINER_WORKER','/usr/local/bin/waffle_gazebo_worker')
        wrapped.extend(('--entrypoint',worker,image,str(world),str(config)))
        cmd=wrapped
    if env.get('WAFFLE_WORKER_AFFINITY','0')!='1':
        return cmd
    allowed=os.sched_getaffinity(0)
    width=int(env.get('WAFFLE_WORKER_CORES','2'))
    reserve=int(env.get('WAFFLE_RESERVE_CORES','2'))
    if width<1 or reserve<0:
        raise ValueError('Invalid worker CPU allocation')
    groups=[]
    for node,cpus in sorted(numa_nodes(allowed).items()):
        cpus=cpus[reserve:] if len(cpus)>reserve else cpus
        for i in range(0,len(cpus),width):
            groups.append((node,cpus[i:i+width]))
    if not groups or not shutil.which('taskset'):
        raise RuntimeError('Worker affinity requires a nonempty Linux cpuset and taskset')
    node,cpus=groups[index%len(groups)]
    # New process receives local affinity before Gazebo/renderer allocates.
    # Preferred NUMA memory policy is opt-in because containers may deny it.
    prefix=['taskset','-c',format_cpus(cpus)]
    if env.get('WAFFLE_NUMA_MEMORY','0')=='1':
        if not shutil.which('numactl'):
            raise RuntimeError('numactl required for preferred memory policy')
        prefix+=['numactl','--preferred='+str(node)]
    return prefix+cmd


def candidate_gpu(slot):
    import json
    if slot.startswith('cpu:'):
        return ''
    mapping=json.loads(os.environ.get('WAFFLE_CANDIDATE_GPU_UUIDS','{}'))
    return mapping.get(slot,slot)


def candidate_cpuset(slot):
    # Explicit device-to-cpuset map supplied by the hardware planner.
    import json
    spec=os.environ.get('WAFFLE_CANDIDATE_CPUSETS')
    if not spec:
        return None
    mapping=json.loads(spec)
    if slot not in mapping:
        raise ValueError('No CPU allocation for candidate '+slot)
    cpus=parse_cpus(mapping[slot])
    if not cpus<=os.sched_getaffinity(0):
        raise ValueError('Candidate CPU allocation outside job cpuset')
    return format_cpus(cpus)
