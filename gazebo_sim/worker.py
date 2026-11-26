
"""Persistent native worker subprocess with bounded transactional IPC.

One Gazebo server per OS process/partition. No ROS bridge or physical topics.
Restart the entire worker at episode reset to clear physics, renderer, motor
and detector state; no synthetic transition is inserted after a failure.
"""
from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import uuid
import mmap
import time
import numpy as np


class Worker:
    def __init__(self,binary,world,config,log,timeout=60.,seed=0):
        self.timeout=timeout
        self.sequence=0
        self.closed=False
        self.lock=threading.Lock()
        self.scan_map=None
        self.scan_fd=None
        self.stats=dict(transactions=0,ipc_wait_s=0.,scan_copy_s=0.,scan_bytes=0)
        self.responses=queue.Queue()
        self.tail=deque(maxlen=25)
        self.log=Path(log).open('w')
        env=os.environ.copy()
        env.pop('WAFFLE_SCAN_FD',None)
        env.update(GZ_PARTITION='waffle-sim-'+uuid.uuid4().hex,
                   ROS_LOCALHOST_ONLY='1',GZ_SIM_RESOURCE_PATH=str(Path(world).parent),
                   OMP_NUM_THREADS='1')
        try:
            transport=env.get('WAFFLE_SCAN_TRANSPORT','json')
            if transport not in ('json','shm'):
                raise ValueError('WAFFLE_SCAN_TRANSPORT must be json or shm')
            if transport=='shm':
                self.scan_fd=os.memfd_create('waffle_scan',os.MFD_CLOEXEC)
                os.ftruncate(self.scan_fd,256*4096*4)
                self.scan_map=mmap.mmap(self.scan_fd,256*4096*4,access=mmap.ACCESS_READ)
                env['WAFFLE_SCAN_FD']=str(self.scan_fd)
            cmd=[str(Path(binary).resolve()),str(Path(world).resolve()),str(Path(config).resolve())]
            from .hardware import worker_command
            cmd=worker_command(cmd,env,int(seed))
            self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.log,text=True,bufsize=1,env=env,start_new_session=True,
                pass_fds=() if self.scan_fd is None else (self.scan_fd,))
        except BaseException:
            self._release_scan()
            self.log.close()
            raise
        self.reader=threading.Thread(target=self._read,daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.proc.stdout:
                if line.startswith('WAFFLE_FRAME '):
                    self.responses.put(json.loads(line[len('WAFFLE_FRAME '):]))
                else:
                    self.tail.append(line.rstrip())
                    self.log.write(line)
                    self.log.flush()
        except Exception as exc:
            self.responses.put({'error':str(exc)})
        finally:
            self.responses.put({'error':'Gazebo worker closed stdout'})

    def step(self,iterations,v,w,humans,stop=False):
        with self.lock:
            return self._step(iterations,v,w,humans,stop)

    def _step(self,iterations,v,w,humans,stop):
        if self.closed:
            raise RuntimeError('Worker closed')
        start=time.perf_counter()
        self.sequence+=1
        msg=dict(op='step',sequence=self.sequence,iterations=int(iterations),v=float(v),w=float(w),
                 humans=humans,stop=bool(stop))
        try:
            self.proc.stdin.write(json.dumps(msg,allow_nan=False)+'\n')
            self.proc.stdin.flush()
            reply=self.responses.get(timeout=self.timeout)
        except (OSError,queue.Empty) as exc:
            self.close()
            raise RuntimeError('Gazebo worker unavailable; rollout aborted') from exc
        if 'error' in reply or reply.get('sequence')!=self.sequence:
            self.close()
            raise RuntimeError('Gazebo transaction failed: '+str(reply.get('error','sequence mismatch')))
        self.stats['ipc_wait_s']+=time.perf_counter()-start
        start=time.perf_counter()
        try:
            self._read_scans(reply)
        except BaseException:
            self.close()
            raise
        self.stats['scan_copy_s']+=time.perf_counter()-start
        self.stats['transactions']+=1
        return reply

    def _read_scans(self,reply):
        end=0
        for frame in reply.get('scans',[]):
            if self.scan_map is None:
                if 'ranges' not in frame:
                    raise RuntimeError('Missing JSON ranges')
                self.stats['scan_bytes']+=4*len(frame['ranges'])
                continue
            offset=frame.pop('range_offset',None)
            count=frame.pop('range_count',None)
            if type(offset) is not int or type(count) is not int or offset!=end or not 1<=count<=4096 or offset+count>256*4096 or 'ranges' in frame:
                raise RuntimeError('Invalid shared scan transaction')
            frame['ranges']=np.frombuffer(self.scan_map,dtype=np.float32,count=count,offset=offset*4).copy()
            end=offset+count
            self.stats['scan_bytes']+=count*4

    def _release_scan(self):
        if self.scan_map is not None:
            self.scan_map.close()
            self.scan_map=None
        if self.scan_fd is not None:
            os.close(self.scan_fd)
            self.scan_fd=None

    def close(self):
        if self.closed:
            return
        self.closed=True
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"op":"close"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except (OSError,subprocess.TimeoutExpired):
                import signal
                try:
                    os.killpg(self.proc.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait()
        self.reader.join(timeout=3)
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.log.close()
        self._release_scan()
