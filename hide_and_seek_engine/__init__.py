# Re-export everything from the C extension so that:
#   import hide_and_seek_engine
#   hide_and_seek_engine.BatchedEnvironment(...)  <-- still works
# while also supporting:
#   from hide_and_seek_engine.env_wrapper import BatchedGridEnv, FeatureType
from ._core import *
from ._core import FeatureType, BatchedEnvironment

# Global-communication mode (smaller state, no expected_obs / last_agent_locations)
from ._core_global import BatchedEnvironment as BatchedEnvironmentGlobal

from .env_wrapper import (
    BatchedGridEnv,
    SARBatchedGridEnv,
    SARGlobalModeEnv,
    SARParallelPettingZooEnv,
    build_terrain_tensor_from_png,
    load_sar_config,
)
