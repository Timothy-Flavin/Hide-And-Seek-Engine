from hide_and_seek_engine.env_wrapper import (
    SARBatchedGridEnv
)

# basic runner to test fps and correctness of the C++ implementation
if __name__ == "__main__":
    env = SARBatchedGridEnv(
        num_envs: int,
        map_png: str,
        tiles_json: str,
        agents_json: str,
        survivors_json: str,
        map_size: int = 32,
        device: str = "cpu",
        seed: int = 42,
        cooperative_rewards: bool = True,
        reward_new_tile: float = 0.05,
        reward_found: float = 2.0,
        reward_saved: float = 20.0,
    )
