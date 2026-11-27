import unittest
from types import SimpleNamespace
from gazebo_sim.worker import Worker


class BrokenPipe:
    def close(self):
        raise BrokenPipeError('closed worker pipe')


class Reader:
    def join(self, timeout=None):
        pass


class WorkerCloseTests(unittest.TestCase):
    def test_broken_worker_pipe_does_not_mask_cleanup(self):
        worker=object.__new__(Worker)
        worker.closed=False
        worker.proc=SimpleNamespace(poll=lambda: 1, stdin=BrokenPipe(), stdout=BrokenPipe())
        worker.reader=Reader()
        worker.log=SimpleNamespace(close=lambda: None)
        worker.scan_map=None
        worker.scan_fd=None
        worker.close()
        self.assertTrue(worker.closed)


if __name__=='__main__':
    unittest.main()
