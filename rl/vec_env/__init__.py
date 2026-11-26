"""Core VecEnv API; legacy Gym/Baselines adapters imported only on demand."""
from .vec_env import AlreadySteppingError, NotSteppingError, VecEnv, VecEnvWrapper, VecEnvObservationWrapper, CloudpickleWrapper

_LAZY = {'DummyVecEnv':'.dummy_vec_env','ShmemVecEnv':'.shmem_vec_env',
         'VecFrameStack':'.vec_frame_stack','VecNormalize':'.vec_normalize',
         'VecExtractDictObs':'.vec_remove_dict_obs'}

def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    from importlib import import_module
    value=getattr(import_module(_LAZY[name],__name__),name)
    globals()[name]=value
    return value

__all__=['AlreadySteppingError','NotSteppingError','VecEnv','VecEnvWrapper',
         'VecEnvObservationWrapper','CloudpickleWrapper',*_LAZY]
