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
from RL.MixedObservationEncoder import MixedObservationEncoder
from RL.checkpoint_utils import CheckpointSaver, variant_name, results_dir_for, module_state


@dataclass
class Args:
    exp_name: str = "custom_dqn"
    run_number: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    centralized: bool = True

    # Ego-centric obs: fixed ego_size x ego_size window centered on each agent.
    # Ego obs is always per-agent, so use it with the decentralized path
    # (--no-centralized).
    ego_view: bool = False
    ego_size: int = 32
    # Path to the level directory (level.png, tiles.json, agents.json, survivors.json).
    level: str = "levels/test_level"
    # Decentralized-only ablation: add a trainable per-agent radio Q-head so
    # agents learn to share observations (use with --no-centralized).
    use_radio: bool = False

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
        self, capacity, num_envs, n_agents, spatial_shape, internal_dim, device, store_radio=False
    ):
        self.capacity = capacity
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.device = device
        self.store_radio = store_radio
        if store_radio:
            self.radio_actions = torch.empty((capacity, n_agents), dtype=torch.int64).contiguous()

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

    def add(self, spatial, internal, next_spatial, next_internal, action, reward, done, radio_action=None):
        batch_size = spatial.shape[0]

        # Convert action to tensor if it isn't already (usually numpy array)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int64)
        if self.store_radio and not isinstance(radio_action, torch.Tensor):
            radio_action = torch.tensor(radio_action, dtype=torch.int64)

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
            if self.store_radio:
                self.radio_actions[idx].copy_(radio_action)
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
            if self.store_radio:
                self.radio_actions[idx1].copy_(radio_action[:part1_len])
                self.radio_actions[idx2].copy_(radio_action[part1_len:])

        self.size = min(self.size + batch_size, self.capacity)
        self.pos = (self.pos + batch_size) % self.capacity

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,))
        if self.store_radio:
            return (
                self.spatial_obs[idxs].to(self.device),
                self.internal_obs[idxs].to(self.device),
                self.next_spatial_obs[idxs].to(self.device),
                self.next_internal_obs[idxs].to(self.device),
                self.actions[idxs].to(self.device),
                self.rewards[idxs].to(self.device),
                self.dones[idxs].to(self.device),
                self.radio_actions[idxs].to(self.device),
            )
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
    def __init__(self, spatial_shape, internal_dim, n_agents, centralized=True, num_actions_per_agent=5, use_radio=False, n_radio_actions=0):
        super().__init__()
        self.n_agents = n_agents
        self.num_actions_per_agent = num_actions_per_agent
        self.spatial_shape = spatial_shape
        self.internal_dim = internal_dim
        self.centralized = centralized
        # Radio ablation: decentralized-only trainable per-agent radio Q-head.
        self.use_radio = use_radio and not centralized
        self.n_radio_actions = n_radio_actions

        self.encoder = MixedObservationEncoder(
            spatial_shape=spatial_shape,
            vector_dim=np.prod(internal_dim) if centralized else internal_dim[1],
            spatial_hidden_dim=128,
            vector_hidden_dim=32,
            output_dim=256,
        )
        # Advantage head width = move dims (+ radio dims for the radio ablation).
        # One fused head emits sum(discrete_dims); the joint VDN Q is
        # V + sum_a adv_move_a[move] + sum_a adv_radio_a[radio]. The move/radio
        # advantages are a zero-copy split of the single head output.
        self.head_out_dim = num_actions_per_agent + (n_radio_actions if self.use_radio else 0)
        self.adv_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.head_out_dim),
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
        elif self.use_radio:
            self.get_action = self.get_action_radio
            self.forward = self._forward_decentralized
        else:
            self.get_action = self._get_action_decentralized
            self.forward = self._forward_decentralized

    def get_action_radio(self, spatial, internal):
        # Decentralized radio ablation: ONE encoder pass, ONE fused advantage head;
        # split the output into move/radio and take each greedy action.
        B = spatial.shape[0]
        x = torch.cat([spatial.view(B * self.n_agents, -1), internal.view(B * self.n_agents, -1)], dim=-1)
        feats = self.encoder(x)
        adv_all = self.adv_heads[0](feats).view(B, self.n_agents, self.head_out_dim)
        move_action = torch.argmax(adv_all[..., :self.num_actions_per_agent], dim=2)
        radio_action = torch.argmax(adv_all[..., self.num_actions_per_agent:], dim=2)
        return move_action, radio_action

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

        # Dueling identifiability: mean-center each agent's advantage over its
        # action dim so V and A cannot drift against each other.
        adv_values = adv_values - adv_values.mean(dim=2, keepdim=True)

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
        # Width is head_out_dim (== num_actions when no radio). Callers split off
        # the radio slice when use_radio is set.
        adv_values = adv_values.view(B, self.n_agents, self.head_out_dim)

        # Dueling identifiability: mean-center each advantage factor over its own
        # action dim. Move and radio are independent heads, so they must be
        # centered separately (centering the concatenation would couple them).
        if self.use_radio:
            move = adv_values[..., :self.num_actions_per_agent]
            radio = adv_values[..., self.num_actions_per_agent:]
            move = move - move.mean(dim=2, keepdim=True)
            radio = radio - radio.mean(dim=2, keepdim=True)
            adv_values = torch.cat([move, radio], dim=2)
        else:
            adv_values = adv_values - adv_values.mean(dim=2, keepdim=True)

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
        map_png=os.path.join(args.level, "level.png"),
        tiles_json=os.path.join(args.level, "tiles.json"),
        agents_json=os.path.join(args.level, "agents.json"),
        survivors_json=os.path.join(args.level, "survivors.json"),
        mode="centralized" if args.centralized else "decentralized",
        requires_state=False,
        device=device,
        ego_view=args.ego_view,
        ego_size=args.ego_size,
    )

    n_agents = env.config.n_agents
    # Single-agent spatial map shape (C, H, W) or (C, ego, ego); robust to ego mode.
    spatial_shape = env.map_spatial_shape
    buffer_spatial_shape = env.obs_spatial_shape
    internal_dim = (env.config.n_agents, env.agent_internal_dim)

    if args.use_radio and args.centralized:
        raise ValueError("--use-radio is a decentralized ablation; run it with --no-centralized")
    # Radio action space: 0 = no broadcast, 1..n_agents-1 = share with a peer.
    n_radio_actions = n_agents
    use_radio = bool(args.use_radio and not args.centralized)

    q_network = torch.compile(
        QNetwork(spatial_shape, internal_dim, n_agents, centralized=args.centralized, use_radio=use_radio, n_radio_actions=n_radio_actions).to(device)
    )
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = torch.compile(
        QNetwork(spatial_shape, internal_dim, n_agents, centralized=args.centralized, use_radio=use_radio, n_radio_actions=n_radio_actions).to(device)
    )
    target_network.load_state_dict(q_network.state_dict())

    rb = CustomReplayBuffer(
        args.buffer_size, args.num_envs, n_agents, buffer_spatial_shape, internal_dim, device, store_radio=use_radio
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

        if use_radio:
            # Per-factor epsilon-greedy; a single combined forward provides both
            # greedy actions when either factor needs one.
            explore_move = random.random() < epsilon
            explore_radio = random.random() < epsilon
            if not (explore_move and explore_radio):
                with torch.no_grad():
                    move_greedy, radio_greedy = q_network.get_action_radio(spatial, internal)
            if explore_move:
                actions_discrete = np.random.randint(0, num_actions_per_agent, size=(args.num_envs, n_agents))
            else:
                actions_discrete = move_greedy.cpu().numpy()
            if explore_radio:
                radio_discrete = np.random.randint(0, n_radio_actions, size=(args.num_envs, n_agents))
            else:
                radio_discrete = radio_greedy.cpu().numpy()
            radio_actions = radio_discrete.astype(np.int32)
        else:
            if random.random() < epsilon:
                actions_discrete = np.random.randint(
                    0, num_actions_per_agent, size=(args.num_envs, n_agents)
                )
            else:
                with torch.no_grad():
                    actions_discrete = q_network.get_action(spatial, internal).cpu().numpy()
            radio_discrete = None
            radio_actions = np.zeros((args.num_envs, n_agents), dtype=np.int32)

        move_actions = ACTION_MAP[actions_discrete]  # Shape: (num_envs, n_agents, 2)

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
            radio_action=radio_discrete,
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
                if use_radio:
                    (
                        b_spatial,
                        b_internal,
                        b_n_spatial,
                        b_n_internal,
                        b_actions,
                        b_rewards,
                        b_dones,
                        b_radio_actions,
                    ) = rb.sample(args.batch_size)
                else:
                    (
                        b_spatial,
                        b_internal,
                        b_n_spatial,
                        b_n_internal,
                        b_actions,
                        b_rewards,
                        b_dones,
                    ) = rb.sample(args.batch_size)

                move_dim = num_actions_per_agent
                with torch.no_grad():
                    target_v, target_adv_all = target_network(b_n_spatial, b_n_internal)
                    if use_radio:
                        # Factored VDN over the single wide head: max each factor.
                        target_move_max, _ = target_adv_all[..., :move_dim].max(dim=2)
                        target_radio_max, _ = target_adv_all[..., move_dim:].max(dim=2)
                        target_q_sum = target_v + target_move_max.sum(dim=1, keepdim=True) + target_radio_max.sum(dim=1, keepdim=True)
                    else:
                        target_adv_max, _ = target_adv_all.max(dim=2)
                        target_q_sum = target_v + target_adv_max.sum(dim=1, keepdim=True)
                    td_target = b_rewards + args.gamma * target_q_sum * (1 - b_dones)

                v, adv_all = q_network(b_spatial, b_internal)
                if use_radio:
                    move_adv = adv_all[..., :move_dim]
                    radio_adv = adv_all[..., move_dim:]
                    adv_taken = move_adv.gather(2, b_actions.unsqueeze(2)).squeeze(2)
                    radio_taken = radio_adv.gather(2, b_radio_actions.unsqueeze(2)).squeeze(2)
                    old_val_sum = v + adv_taken.sum(dim=1, keepdim=True) + radio_taken.sum(dim=1, keepdim=True)
                else:
                    adv_taken = adv_all.gather(2, b_actions.unsqueeze(2)).squeeze(2)
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

    # Plot episodic returns at the end (results namespaced per level + variant)
    level_name = os.path.basename(os.path.normpath(args.level))
    # Compositional variant name, e.g. decentralized_ego_radio.
    if args.centralized:
        variant = "centralized"
    else:
        variant = "decentralized"
        if args.ego_view:
            variant += "_ego"
        if use_radio:
            variant += "_radio"
    results_dir = os.path.join("experiments/results", level_name)
    os.makedirs(results_dir, exist_ok=True)
    base_name = f"dqn_{variant}_episodic_returns_run_{args.run_number}"
    np.save(os.path.join(results_dir, f"{base_name}.npy"), np.array(episodic_returns))

    plt.figure()
    plt.plot(episodic_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Episodic Returns (Run {args.run_number})")
    plt.savefig(os.path.join(results_dir, f"{base_name}.png"))
    print(
        f"End of training. Plotted {len(episodic_returns)} episodes to '{os.path.join(results_dir, base_name)}.png'."
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
        plt.savefig(os.path.join(results_dir, f"{base_name}_smoothed.png"))
        print(f"Plotted EMA-smoothed episodic returns to '{os.path.join(results_dir, base_name)}_smoothed.png'.")

    plt.figure()
    plt.plot(q_losses)
    plt.xlabel("Update Step")
    plt.ylabel("Q Loss")
    plt.title(f"Q Network Loss (Run {args.run_number})")
    plt.savefig(os.path.join(results_dir, f"dqn_{variant}_q_losses_run_{args.run_number}.png"))
    print(f"Plotted {len(q_losses)} loss values to '{results_dir}/dqn_{variant}_q_losses_run_{args.run_number}.png'.")
