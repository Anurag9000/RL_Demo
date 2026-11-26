
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


class Worker:
    def __init__(self,binary,world,config,log,timeout=60.,seed=0):
        self.timeout=timeout
        self.sequence=0
        self.responses=queue.Queue()
        self.tail=deque(maxlen=25)
        self.log=Path(log).open('w')
        env=os.environ.copy()
        env.update(GZ_PARTITION='waffle-sim-'+uuid.uuid4().hex,
                   ROS_LOCALHOST_ONLY='1',GZ_SIM_RESOURCE_PATH=str(Path(world).parent),
                   OMP_NUM_THREADS='1')
        try:
            self.proc=subprocess.Popen([str(Path(binary).resolve()),str(Path(world).resolve()),
                str(Path(config).resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.log,text=True,bufsize=1,env=env,start_new_session=True)
        except BaseException:
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
        return reply

    def close(self):
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
