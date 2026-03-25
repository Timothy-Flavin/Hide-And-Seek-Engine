from hide_and_seek_engine.env_wrapper import (
    SARBatchedGridEnv
)

# basic runner to test fps and correctness of the C++ implementation
if __name__ == "__main__":
    env = SARBatchedGridEnv(
        num_envs=8,
        map_png="test_level/level.png",
        tiles_json="test_level/tiles.json",
        agents_json="test_level/agents.json",
        survivors_json="test_level/survivors.json",
        map_size=32,
        device="cpu",
        seed=42,
        cooperative_rewards = True,
        reward_new_tile = 0.05,
        reward_found = 2.0,
        reward_saved = 20.0,
    )

    for i in range(10):
        obs = env.reset()
        done = [False] * env.num_envs
        while not all(done):
            actions = [env.action_space.sample() for _ in range(env.num_envs)]
            obs, rewards, terminated, truncated, info = env.step(actions)
            env.render()
