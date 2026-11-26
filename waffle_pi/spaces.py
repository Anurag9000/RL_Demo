"""Only space metadata is needed by the tensor backend."""
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
