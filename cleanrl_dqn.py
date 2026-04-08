import os
import random
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

torch.set_float32_matmul_precision("high")

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
from MixedObservationEncoder import MixedObservationEncoder


@dataclass
class Args:
    exp_name: str = "custom_dqn"
    run_number: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    centralized: bool = True

    total_timesteps: int = 10000000
    learning_rate: float = 2.5e-4
    num_envs: int = 128
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 2048
    batch_size: int = 1024
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.2
    learning_starts: int = 1000
    train_frequency: int = 128


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class CustomReplayBuffer:
    def __init__(
        self, capacity, num_envs, n_agents, spatial_shape, internal_dim, device
    ):
        self.capacity = capacity
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.device = device

        self.spatial_obs = torch.empty(
            (capacity, *spatial_shape), dtype=torch.float32, pin_memory=True
        ).contiguous()
        self.internal_obs = torch.empty(
            (capacity, *internal_dim), dtype=torch.float32, pin_memory=True
        ).contiguous()

        self.next_spatial_obs = torch.empty_like(self.spatial_obs).contiguous()
        self.next_internal_obs = torch.empty_like(self.internal_obs).contiguous()

        self.actions = torch.empty((capacity, n_agents), dtype=torch.int64).contiguous()
        # Summed reward across agents per env for VDN
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32).contiguous()
        self.dones = torch.empty((capacity, 1), dtype=torch.float32).contiguous()

        self.pos = 0
        self.size = 0

    def add(self, spatial, internal, next_spatial, next_internal, action, reward, done):
        batch_size = spatial.shape[0]

        # Convert action to tensor if it isn't already (usually numpy array)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int64)

        # reward and done are already tensors from the env
        summed_reward = reward.sum(dim=1, keepdim=True)
        done_col = done.unsqueeze(1) if done.dim() == 1 else done

        if self.pos + batch_size <= self.capacity:
            idx = slice(self.pos, self.pos + batch_size)

            self.spatial_obs[idx].copy_(spatial)
            self.internal_obs[idx].copy_(internal)
            self.next_spatial_obs[idx].copy_(next_spatial)
            self.next_internal_obs[idx].copy_(next_internal)
            self.actions[idx].copy_(action)
            self.rewards[idx].copy_(summed_reward)
            self.dones[idx].copy_(done_col)
        else:
            # Handle wrap around
            part1_len = self.capacity - self.pos
            part2_len = batch_size - part1_len

            idx1 = slice(self.pos, self.capacity)
            idx2 = slice(0, part2_len)

            self.spatial_obs[idx1].copy_(spatial[:part1_len])
            self.internal_obs[idx1].copy_(internal[:part1_len])
            self.next_spatial_obs[idx1].copy_(next_spatial[:part1_len])
            self.next_internal_obs[idx1].copy_(next_internal[:part1_len])
            self.actions[idx1].copy_(action[:part1_len])
            self.rewards[idx1].copy_(summed_reward[:part1_len])
            self.dones[idx1].copy_(done_col[:part1_len])

            self.spatial_obs[idx2].copy_(spatial[part1_len:])
            self.internal_obs[idx2].copy_(internal[part1_len:])
            self.next_spatial_obs[idx2].copy_(next_spatial[part1_len:])
            self.next_internal_obs[idx2].copy_(next_internal[part1_len:])
            self.actions[idx2].copy_(action[part1_len:])
            self.rewards[idx2].copy_(summed_reward[part1_len:])
            self.dones[idx2].copy_(done_col[part1_len:])

        self.size = min(self.size + batch_size, self.capacity)
        self.pos = (self.pos + batch_size) % self.capacity

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,))
        return (
            self.spatial_obs[idxs].to(self.device),
            self.internal_obs[idxs].to(self.device),
            self.next_spatial_obs[idxs].to(self.device),
            self.next_internal_obs[idxs].to(self.device),
            self.actions[idxs].to(self.device),
            self.rewards[idxs].to(self.device),
            self.dones[idxs].to(self.device),
        )


class QNetwork(nn.Module):
    def __init__(self, spatial_shape, internal_dim, n_agents, centralized=True, num_actions_per_agent=5):
        super().__init__()
        self.n_agents = n_agents
        self.num_actions_per_agent = num_actions_per_agent
        self.spatial_shape = spatial_shape
        self.internal_dim = internal_dim
        self.centralized = centralized

        self.encoder = MixedObservationEncoder(
            spatial_shape=spatial_shape,
            vector_dim=np.prod(internal_dim) if centralized else internal_dim[1],
            spatial_hidden_dim=128,
            vector_hidden_dim=32,
            output_dim=256,
        )
        self.adv_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_actions_per_agent),
                )
                for _ in range(n_agents if centralized else 1)
            ]
        )

        self.v_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                )

        if centralized:
            self.get_action = self._get_action_centralized
            self.forward = self._forward_centralized
        else:
            self.get_action = self._get_action_decentralized
            self.forward = self._forward_decentralized

    def _get_action_centralized(self, spatial, internal):
        B = spatial.shape[0]
        spatial_flat = spatial.view(B, -1)
        internal_flat = internal.view(B, -1)
        x = torch.cat([spatial_flat, internal_flat], dim=-1)

        feats = self.encoder(x)
        adv_values = torch.stack([head(feats) for head in self.adv_heads], dim=1) # [B, n_agents, num_actions]
        return torch.argmax(adv_values, dim=2)

    def _get_action_decentralized(self, spatial, internal):
        B = spatial.shape[0]
        spatial_flat = spatial.view(B * self.n_agents, -1)
        internal_flat = internal.view(B * self.n_agents, -1)
        x = torch.cat([spatial_flat, internal_flat], dim=-1)

        feats = self.encoder(x)
        adv_values = self.adv_heads[0](feats)
        adv_values = adv_values.view(B, self.n_agents, self.num_actions_per_agent)
        return torch.argmax(adv_values, dim=2)

    def _forward_centralized(self, spatial, internal):
        B = spatial.shape[0]
        spatial_flat = spatial.view(B, -1)
        internal_flat = internal.view(B, -1)
        x = torch.cat([spatial_flat, internal_flat], dim=-1)

        feats = self.encoder(x)

        v_value = self.v_head(feats) # [B, 1]
        adv_values = torch.stack([head(feats) for head in self.adv_heads], dim=1) # [B, n_agents, num_actions]

        return v_value, adv_values

    def _forward_decentralized(self, spatial, internal):
        B = spatial.shape[0]
        spatial_flat = spatial.view(B * self.n_agents, -1)
        internal_flat = internal.view(B * self.n_agents, -1)
        x = torch.cat([spatial_flat, internal_flat], dim=-1)

        feats = self.encoder(x)

        v_value = self.v_head(feats) # [B*n_agents, 1]
        v_value = v_value.view(B, self.n_agents, 1).sum(dim=1) # Sum over agents for VDN

        adv_values = self.adv_heads[0](feats)
        adv_values = adv_values.view(B, self.n_agents, self.num_actions_per_agent)

        return v_value, adv_values


ACTION_MAP = np.array(
    [
        [0.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
    ],
    dtype=np.float32,
)

if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.run_number)
    np.random.seed(args.run_number)
    torch.manual_seed(args.run_number)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = SARBatchedGridEnv(
        num_envs=args.num_envs,
        map_png="test_level/level.png",
        tiles_json="test_level/tiles.json",
        agents_json="test_level/agents.json",
        survivors_json="test_level/survivors.json",
        mode="centralized" if args.centralized else "decentralized",
        requires_state=False,
        device=device,
    )

    n_agents = env.config.n_agents
    spatial_shape = (env.spatial_channels, env.config.height, env.config.width)
    buffer_spatial_shape = spatial_shape if args.centralized else (n_agents, *spatial_shape)
    internal_dim = (env.config.n_agents, env.agent_internal_dim)

    q_network = torch.compile(
        QNetwork(spatial_shape, internal_dim, n_agents, centralized=args.centralized).to(device)
    )
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = torch.compile(
        QNetwork(spatial_shape, internal_dim, n_agents, centralized=args.centralized).to(device)
    )
    target_network.load_state_dict(q_network.state_dict())

    rb = CustomReplayBuffer(
        args.buffer_size, args.num_envs, n_agents, buffer_spatial_shape, internal_dim, device
    )

    start_time = time.time()
    last_log_time = start_time
    last_update_time = start_time
    step_count = 0
    update_count = 0
    cached_logical_steps = 0
    num_actions_per_agent = 5
    episodic_returns = []
    n_episodes_passed = 0
    logical_steps_since_last_record = 0
    LOGICAL_STEPS_PER_RECORD = 10000
    q_losses = []

    obs = env._get_obs_dict()
    spatial = obs["spatial"]
    internal = obs["internal"]

    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)

    for iteration in range(args.total_timesteps // args.num_envs):
        global_step = iteration * args.num_envs

        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        if random.random() < epsilon:
            actions_discrete = np.random.randint(
                0, num_actions_per_agent, size=(args.num_envs, n_agents)
            )
        else:
            with torch.no_grad():
                actions_discrete_t = q_network.get_action(spatial, internal)
                actions_discrete = actions_discrete_t.cpu().numpy()

        move_actions = ACTION_MAP[actions_discrete]  # Shape: (num_envs, n_agents, 2)
        radio_actions = np.zeros((args.num_envs, n_agents), dtype=np.int32)

        next_obs, rewards, terminations, truncations, infos = env.step(
            move_actions, radio_actions
        )

        n_spatial = next_obs["spatial"]
        n_internal = next_obs["internal"]

        # Sum rewards over all agents for each environment, yielding an array of shape (num_envs,)
        env_rewards = rewards.sum(dim=1).cpu().numpy()
        episode_rewards += env_rewards

        rb.add(
            spatial,
            internal,
            n_spatial,
            n_internal,
            actions_discrete,
            rewards,
            terminations,
        )

        spatial = n_spatial
        internal = n_internal

        # Track logical steps for episodic return recording
        logical_steps_since_last_record += args.num_envs

        if terminations.any() or truncations.any():
            for e in range(args.num_envs):
                if terminations[e] or truncations[e]:
                    n_episodes_passed += 1
                    episodic_returns.append(episode_rewards[e])
                    episode_rewards[e] = 0.0

            obs = env._get_obs_dict()
            spatial = obs["spatial"]
            internal = obs["internal"]

        # --- Steps/sec and model updates/sec tracking ---
        step_count += args.num_envs
        cached_logical_steps += args.num_envs

        # Model update every 4 logical steps (across all envs)
        while cached_logical_steps >= args.train_frequency:
            if global_step > args.learning_starts:
                (
                    b_spatial,
                    b_internal,
                    b_n_spatial,
                    b_n_internal,
                    b_actions,
                    b_rewards,
                    b_dones,
                ) = rb.sample(args.batch_size)

                with torch.no_grad():
                    target_v, target_adv = target_network(b_n_spatial, b_n_internal)
                    target_adv_max, _ = target_adv.max(dim=2)
                    target_q_sum = target_v + target_adv_max.sum(dim=1, keepdim=True)
                    td_target = b_rewards + args.gamma * target_q_sum * (1 - b_dones)

                v, adv = q_network(b_spatial, b_internal)
                adv_taken = adv.gather(2, b_actions.unsqueeze(2)).squeeze(2)
                old_val_sum = v + adv_taken.sum(dim=1, keepdim=True)

                loss = F.mse_loss(td_target, old_val_sum)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                q_losses.append(loss.item())
                update_count += 1
            cached_logical_steps -= args.train_frequency

        # --- Logging steps/sec and updates/sec every 1000 env steps ---
        if step_count >= args.total_timesteps//100:
            now = time.time()
            elapsed = now - last_log_time
            steps_per_sec = step_count / elapsed
            updates_per_sec = update_count / elapsed if elapsed > 0 else 0.0
            print(
                f"Steps/sec: {steps_per_sec:.2f}, up/sec: {updates_per_sec:.2f} qloss: {sum(q_losses[-100:])/100}, return: {sum(episodic_returns[-100:])/100} left: {(args.total_timesteps-global_step)/steps_per_sec}s"
            )
            last_log_time = now
            step_count = 0
            update_count = 0
        # Target network update
        if iteration % max(1, args.target_network_frequency // args.num_envs) == 0:
            for target_network_param, q_network_param in zip(
                target_network.parameters(), q_network.parameters()
            ):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data
                    + (1.0 - args.tau) * target_network_param.data
                )

    # Plot episodic returns at the end
    os.makedirs("results", exist_ok=True)
    base_name = f"dqn_episodic_returns_{'centralized' if args.centralized else 'individual'}_run_{args.run_number}"
    np.save(f"results/{base_name}.npy", np.array(episodic_returns))

    plt.figure()
    plt.plot(episodic_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Episodic Returns (Run {args.run_number})")
    plt.savefig(f"results/{base_name}.png")
    print(
        f"End of training. Plotted {len(episodic_returns)} episodes to 'results/{base_name}.png'."
    )

    # Plot smoothed episodic returns using EMA (0.99)
    if len(episodic_returns) > 0:
        ema_returns = [episodic_returns[0]]
        for r in episodic_returns[1:]:
            ema_returns.append(0.99 * ema_returns[-1] + 0.01 * r)
        plt.figure()
        plt.plot(ema_returns)
        plt.xlabel("Episode")
        plt.ylabel("EMA Return (0.99)")
        plt.title(f"Smoothed Episodic Returns (EMA 0.99) (Run {args.run_number})")
        plt.savefig(f"results/{base_name}_smoothed.png")
        plt.savefig(f"results/dqn_episodic_returns_ema99_run_{args.run_number}.png")
        print(f"Plotted EMA-smoothed episodic returns to 'results/dqn_episodic_returns_ema99_run_{args.run_number}.png'.")

    plt.figure()
    plt.plot(q_losses)
    plt.xlabel("Update Step")
    plt.ylabel("Q Loss")
    plt.title(f"Q Network Loss (Run {args.run_number})")
    plt.savefig(f"results/dqn_q_losses_run_{args.run_number}.png")
    print(f"Plotted {len(q_losses)} loss values to 'results/dqn_q_losses_run_{args.run_number}.png'.")
