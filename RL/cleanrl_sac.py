import os
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision("high")

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
from RL.MixedObservationEncoder import MixedObservationEncoder
from RL.checkpoint_utils import CheckpointSaver, variant_name, results_dir_for, module_state


@dataclass
class Args:
    exp_name: str = "custom_sac"
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    run_number: int = 1
    """the run number for this experiment"""
    centralized: bool = True
    """whether to use centralized or decentralized execution"""
    ego_view: bool = False
    """ego-centric obs: fixed window centered on each agent (use with --no-centralized)"""
    ego_size: int = 32
    """side length of the ego-centric obs window when ego_view is set"""
    level: str = "levels/test_level"
    """path to the level directory (expects level.png, tiles.json, agents.json, survivors.json)"""
    use_radio: bool = False
    """decentralized-only ablation: add a trainable per-agent radio head (discrete bins) so agents learn to share observations (use with --no-centralized)"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    num_envs: int = 128
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005 for soft updates)"""
    batch_size: int = 1024
    """the batch size of sample from the reply memory"""
    learning_starts: int = 4096
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network optimizer"""
    train_frequency: int = 128
    """how many logical steps between model updates"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.5
    """coefficient for scaling the autotune entropy target"""


class CustomReplayBuffer:
    def __init__(self, capacity, num_envs, n_agents, spatial_shape, internal_dim, device, store_radio=False):
        self.capacity = capacity
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.device = device
        self.store_radio = store_radio

        self.spatial_obs = torch.empty((capacity, *spatial_shape), dtype=torch.float32, pin_memory=True).contiguous()
        self.internal_obs = torch.empty((capacity, *internal_dim), dtype=torch.float32, pin_memory=True).contiguous()

        self.next_spatial_obs = torch.empty_like(self.spatial_obs).contiguous()
        self.next_internal_obs = torch.empty_like(self.internal_obs).contiguous()

        self.actions = torch.empty((capacity, n_agents), dtype=torch.int64).contiguous()
        if store_radio:
            self.radio_actions = torch.empty((capacity, n_agents), dtype=torch.int64).contiguous()
        # Summed reward across agents per env for VDN
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32).contiguous()
        self.dones = torch.empty((capacity, 1), dtype=torch.float32).contiguous()

        self.pos = 0
        self.size = 0

    def add(self, spatial, internal, next_spatial, next_internal, action, reward, done, radio_action=None):
        batch_size = spatial.shape[0]

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int64)
        if self.store_radio and not isinstance(radio_action, torch.Tensor):
            radio_action = torch.tensor(radio_action, dtype=torch.int64)

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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, spatial_shape, internal_dim, n_agents, num_actions_per_agent=5, centralized=True, use_radio=False, n_radio_actions=0):
        super().__init__()
        self.n_agents = n_agents
        self.num_actions_per_agent = num_actions_per_agent
        self.centralized = centralized
        # Radio ablation: decentralized-only per-agent radio advantage head (bins).
        self.use_radio = use_radio and not centralized
        self.n_radio_actions = n_radio_actions

        self.encoder = MixedObservationEncoder(
            spatial_shape=spatial_shape,
            vector_dim=np.prod(internal_dim) if centralized else np.prod(internal_dim) // n_agents,
            spatial_hidden_dim=128,
            vector_hidden_dim=32,
            output_dim=256,
        )
        # Advantage head width = move dims (+ radio dims for the radio ablation).
        # One fused head per agent-slot emits sum(discrete_dims); the move/radio
        # advantages are a zero-copy split of the single head output.
        self.head_out_dim = num_actions_per_agent + (n_radio_actions if self.use_radio else 0)
        self.num_adv_heads = n_agents if centralized else 1
        self.adv_heads = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(256, 128)),
                nn.ReLU(),
                layer_init(nn.Linear(128, self.head_out_dim))
            ) for _ in range(self.num_adv_heads)
        ])

        self.v_head = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1))
        )

        if centralized:
            self.forward = self._forward_centralized
        elif self.use_radio:
            self.forward = self._forward_decentralized_radio
        else:
            self.forward = self._forward_decentralized

    def _forward_decentralized_radio(self, spatial, internal, move_log_pi=None, radio_log_pi=None, alpha=None):
        # Decentralized radio ablation: ONE encoder pass, ONE fused advantage head;
        # split into move/radio factors, each mean-centered (and entropy-adjusted
        # when log-probs are supplied) exactly like the base move advantage.
        B = spatial.shape[0]
        A = spatial.shape[1] if spatial.dim() == 5 else self.n_agents
        spatial = spatial.view(B * A, *spatial.shape[-3:])
        internal = internal.view(B * A, -1)
        x = torch.cat([spatial.view(B * A, -1), internal.view(B * A, -1)], dim=-1)
        feats = self.encoder(x)

        v_value = self.v_head(feats).view(B, A, 1).sum(dim=1)  # [B, 1]
        adv_all = self.adv_heads[0](feats).view(B, A, self.head_out_dim)
        adv_move = adv_all[..., :self.num_actions_per_agent]
        adv_radio = adv_all[..., self.num_actions_per_agent:]

        if move_log_pi is not None and alpha is not None:
            adv_move = adv_move - alpha * move_log_pi
        adv_move = adv_move - adv_move.mean(dim=2, keepdim=True)

        if radio_log_pi is not None and alpha is not None:
            adv_radio = adv_radio - alpha * radio_log_pi
        adv_radio = adv_radio - adv_radio.mean(dim=2, keepdim=True)

        return v_value, adv_move, adv_radio

    def _forward_centralized(self, spatial, internal, log_pi=None, alpha=None):
        B = spatial.shape[0]
        B_eff = B
            
        x = torch.cat([spatial.view(B_eff, -1), internal.view(B_eff, -1)], dim=-1)
        feats = self.encoder(x)
        
        v_value = self.v_head(feats) # [B_eff, 1]
        adv_values = torch.stack([head(feats) for head in self.adv_heads], dim=1) # [B_eff, num_heads, num_actions]
        
        if log_pi is not None and alpha is not None:
            adv_values = adv_values - alpha * log_pi
            
        adv_values = adv_values - adv_values.mean(dim=2, keepdim=True)
        return v_value, adv_values  # [B, 1], [B, n_agents, num_actions]

    def _forward_decentralized(self, spatial, internal, log_pi=None, alpha=None):
        B = spatial.shape[0]
        A = spatial.shape[1] if spatial.dim() == 5 else self.n_agents
        spatial = spatial.view(B * A, *spatial.shape[-3:])
        internal = internal.view(B * A, -1)
        B_eff = B * A
            
        x = torch.cat([spatial.view(B_eff, -1), internal.view(B_eff, -1)], dim=-1)
        feats = self.encoder(x)
        
        v_value = self.v_head(feats) # [B_eff, 1]
        adv_values = torch.stack([head(feats) for head in self.adv_heads], dim=1) # [B_eff, num_heads, num_actions]
        
        v_value = v_value.view(B, A, 1).sum(dim=1) # [B, 1]
        adv_values = adv_values.view(B, A, self.num_actions_per_agent) # [B, n_agents, num_actions]
            
        if log_pi is not None and alpha is not None:
            adv_values = adv_values - alpha * log_pi
            
        adv_values = adv_values - adv_values.mean(dim=2, keepdim=True)
        return v_value, adv_values  # [B, 1], [B, n_agents, num_actions]


class Actor(nn.Module):
    def __init__(self, spatial_shape, internal_dim, n_agents, num_actions_per_agent=5, centralized=True, use_radio=False, n_radio_actions=0):
        super().__init__()
        self.n_agents = n_agents
        self.num_actions_per_agent = num_actions_per_agent
        self.centralized = centralized
        # Radio ablation: decentralized-only per-agent radio policy head (bins).
        self.use_radio = use_radio and not centralized
        self.n_radio_actions = n_radio_actions

        self.encoder = MixedObservationEncoder(
            spatial_shape=spatial_shape,
            vector_dim=np.prod(internal_dim) if centralized else np.prod(internal_dim) // n_agents,
            spatial_hidden_dim=128,
            vector_hidden_dim=32,
            output_dim=256,
        )
        # Fused policy head of width (move + radio) for the radio ablation; the
        # logits are zero-copy split into the two factors (single kernel).
        self.head_out_dim = num_actions_per_agent + (n_radio_actions if self.use_radio else 0)
        self.num_actor_heads = n_agents if centralized else 1
        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(256, 128)),
                nn.ReLU(),
                layer_init(nn.Linear(128, self.head_out_dim), std=0.01)
            ) for _ in range(self.num_actor_heads)
        ])

        if centralized:
            self.forward = self._forward_centralized
        else:
            self.forward = self._forward_decentralized

    def get_action_radio(self, spatial, internal):
        # Decentralized radio ablation: ONE encoder pass, ONE fused head; split the
        # logits into move/radio and return (action, log_prob, probs) for each.
        B = spatial.shape[0]
        A = spatial.shape[1] if spatial.dim() == 5 else self.n_agents
        s = spatial.view(B * A, *spatial.shape[-3:])
        it = internal.view(B * A, -1)
        x = torch.cat([s.view(B * A, -1), it.view(B * A, -1)], dim=-1)
        feats = self.encoder(x)
        logits_all = self.actor_heads[0](feats).view(B, A, self.head_out_dim)
        move_logits = logits_all[..., :self.num_actions_per_agent]
        radio_logits = logits_all[..., self.num_actions_per_agent:]

        move_dist = Categorical(logits=move_logits)
        radio_dist = Categorical(logits=radio_logits)
        move_action = move_dist.sample()
        radio_action = radio_dist.sample()
        move_log_prob = F.log_softmax(move_logits, dim=2)
        radio_log_prob = F.log_softmax(radio_logits, dim=2)
        return (
            move_action, move_log_prob, move_dist.probs,
            radio_action, radio_log_prob, radio_dist.probs,
        )

    def _forward_centralized(self, spatial, internal):
        B = spatial.shape[0]
        B_eff = B
            
        x = torch.cat([spatial.view(B_eff, -1), internal.view(B_eff, -1)], dim=-1)
        feats = self.encoder(x)
        logits = torch.stack([head(feats) for head in self.actor_heads], dim=1)
            
        return logits  # [B, n_agents, num_actions]

    def _forward_decentralized(self, spatial, internal):
        B = spatial.shape[0]
        A = spatial.shape[1] if spatial.dim() == 5 else self.n_agents
        spatial = spatial.view(B * A, *spatial.shape[-3:])
        internal = internal.view(B * A, -1)
        B_eff = B * A
            
        x = torch.cat([spatial.view(B_eff, -1), internal.view(B_eff, -1)], dim=-1)
        feats = self.encoder(x)
        logits = torch.stack([head(feats) for head in self.actor_heads], dim=1)
        
        logits = logits.view(B, A, self.num_actions_per_agent)
            
        return logits  # [B, n_agents, num_actions]

    def get_action(self, spatial, internal):
        logits = self(spatial, internal)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        
        # Action probabilities for calculating the exact expectation in discrete SAC
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=2)
        
        return action, log_prob, action_probs


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
    run_name = f"GridWorld__{args.exp_name}__{args.run_number}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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

    # After resetting the env, check the actual shapes of observations for buffer init
    dummy_obs = env._get_obs_dict()
    spatial_shape = dummy_obs["spatial"].shape[1:]  # (C, H, W), (A, C, H, W), or ego (A, C, S, S)
    internal_dim = dummy_obs["internal"].shape[1:]  # (A, D)

    # Single-agent spatial map shape fed to the encoders; robust to ego mode
    # (in ego mode obs is always per-agent, including centralized).
    single_spatial_shape = env.map_spatial_shape
    
    num_actions_per_agent = 5

    if args.use_radio and args.centralized:
        raise ValueError("--use-radio is a decentralized ablation; run it with --no-centralized")
    # Radio action space (bins): 0 = no broadcast, 1..n_agents-1 = share with a peer.
    n_radio_actions = n_agents
    use_radio = bool(args.use_radio and not args.centralized)

    actor = torch.compile(Actor(single_spatial_shape, internal_dim, n_agents, num_actions_per_agent, args.centralized, use_radio, n_radio_actions).to(device))
    qf1 = torch.compile(SoftQNetwork(single_spatial_shape, internal_dim, n_agents, num_actions_per_agent, args.centralized, use_radio, n_radio_actions).to(device))
    qf2 = torch.compile(SoftQNetwork(single_spatial_shape, internal_dim, n_agents, num_actions_per_agent, args.centralized, use_radio, n_radio_actions).to(device))
    qf1_target = torch.compile(SoftQNetwork(single_spatial_shape, internal_dim, n_agents, num_actions_per_agent, args.centralized, use_radio, n_radio_actions).to(device))
    qf2_target = torch.compile(SoftQNetwork(single_spatial_shape, internal_dim, n_agents, num_actions_per_agent, args.centralized, use_radio, n_radio_actions).to(device))
    
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(num_actions_per_agent))
        log_alpha = torch.empty(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    rb = CustomReplayBuffer(
        args.buffer_size, args.num_envs, n_agents, spatial_shape, internal_dim, device, store_radio=use_radio
    )

    start_time = time.time()
    last_log_time = start_time
    cached_logical_steps = 0
    step_count = 0
    update_count = 0

    obs = env._get_obs_dict()
    spatial = obs["spatial"]
    internal = obs["internal"]

    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    episodic_returns = []

    for iteration in range(args.total_timesteps // args.num_envs):
        global_step = iteration * args.num_envs

        if use_radio:
            if global_step < args.learning_starts:
                actions_discrete = np.random.randint(0, num_actions_per_agent, size=(args.num_envs, n_agents))
                radio_discrete = np.random.randint(0, n_radio_actions, size=(args.num_envs, n_agents))
            else:
                with torch.no_grad():
                    # Single forward pass produces both factored actions.
                    move_t, _, _, radio_t, _, _ = actor.get_action_radio(spatial, internal)
                actions_discrete = move_t.cpu().numpy()
                radio_discrete = radio_t.cpu().numpy()
            radio_actions = radio_discrete.astype(np.int32)
        else:
            if global_step < args.learning_starts:
                actions_discrete = np.random.randint(0, num_actions_per_agent, size=(args.num_envs, n_agents))
            else:
                with torch.no_grad():
                    actions_discrete_t, _, _ = actor.get_action(spatial, internal)
                    actions_discrete = actions_discrete_t.cpu().numpy()
            radio_discrete = None
            radio_actions = np.zeros((args.num_envs, n_agents), dtype=np.int32)

        move_actions = ACTION_MAP[actions_discrete]

        next_obs, rewards, terminations, truncations, infos = env.step(move_actions, radio_actions)

        n_spatial = next_obs["spatial"]
        n_internal = next_obs["internal"]

        env_rewards = rewards.sum(dim=1).cpu().numpy()
        episode_rewards += env_rewards
        episode_lengths += 1

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

        if terminations.any() or truncations.any():
            for e in range(args.num_envs):
                if terminations[e] or truncations[e]:
                    writer.add_scalar("charts/episodic_return", episode_rewards[e], global_step)
                    writer.add_scalar("charts/episodic_length", episode_lengths[e], global_step)
                    episodic_returns.append(episode_rewards[e])
                    episode_rewards[e] = 0.0
                    episode_lengths[e] = 0

            obs_dict = env._get_obs_dict()
            spatial = obs_dict["spatial"]
            internal = obs_dict["internal"]

        # --- Training Loop ---
        step_count += args.num_envs
        cached_logical_steps += args.num_envs

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

                # CRITIC TRAINING
                if use_radio:
                    with torch.no_grad():
                        # One actor pass + one pass per target Q-net (each returns
                        # both factored advantages from a single fused head).
                        (_, next_move_log_pi, next_move_probs,
                         _, next_radio_log_pi, next_radio_probs) = actor.get_action_radio(b_n_spatial, b_n_internal)
                        v_target1, adv_move_t1, adv_radio_t1 = qf1_target(b_n_spatial, b_n_internal, next_move_log_pi, next_radio_log_pi, alpha)
                        v_target2, adv_move_t2, adv_radio_t2 = qf2_target(b_n_spatial, b_n_internal, next_move_log_pi, next_radio_log_pi, alpha)

                        exp1 = (next_move_probs * adv_move_t1).sum(dim=2) + (next_radio_probs * adv_radio_t1).sum(dim=2)
                        exp2 = (next_move_probs * adv_move_t2).sum(dim=2) + (next_radio_probs * adv_radio_t2).sum(dim=2)
                        expected_q1_sum = v_target1 + exp1.sum(dim=1, keepdim=True)
                        expected_q2_sum = v_target2 + exp2.sum(dim=1, keepdim=True)

                        min_expected_q = torch.min(expected_q1_sum, expected_q2_sum)
                        next_q_value = b_rewards + args.gamma * (1 - b_dones) * min_expected_q

                    # Taken-action joint Q from a single pass per net; split factors.
                    v1, adv_move1, adv_radio1 = qf1(b_spatial, b_internal)
                    v2, adv_move2, adv_radio2 = qf2(b_spatial, b_internal)
                    adv1_a = adv_move1.gather(2, b_actions.unsqueeze(2)).squeeze(2)
                    adv2_a = adv_move2.gather(2, b_actions.unsqueeze(2)).squeeze(2)
                    radio1_a = adv_radio1.gather(2, b_radio_actions.unsqueeze(2)).squeeze(2)
                    radio2_a = adv_radio2.gather(2, b_radio_actions.unsqueeze(2)).squeeze(2)
                    qf1_a_values_sum = v1 + adv1_a.sum(dim=1, keepdim=True) + radio1_a.sum(dim=1, keepdim=True)
                    qf2_a_values_sum = v2 + adv2_a.sum(dim=1, keepdim=True) + radio2_a.sum(dim=1, keepdim=True)
                else:
                    with torch.no_grad():
                        _, next_state_log_pi, next_state_action_probs = actor.get_action(b_n_spatial, b_n_internal)
                        v_target1, adv_target1 = qf1_target(b_n_spatial, b_n_internal, next_state_log_pi, alpha)
                        v_target2, adv_target2 = qf2_target(b_n_spatial, b_n_internal, next_state_log_pi, alpha)

                        # Compute expected advantages for each network
                        exp_adv1 = (next_state_action_probs * adv_target1).sum(dim=2)  # [B, n_agents]
                        exp_adv2 = (next_state_action_probs * adv_target2).sum(dim=2)  # [B, n_agents]

                        # Sum to get total expected Q values
                        expected_q1_sum = v_target1 + exp_adv1.sum(dim=1, keepdim=True)  # [B, 1]
                        expected_q2_sum = v_target2 + exp_adv2.sum(dim=1, keepdim=True)  # [B, 1]

                        min_expected_q = torch.min(expected_q1_sum, expected_q2_sum)
                        next_q_value = b_rewards + args.gamma * (1 - b_dones) * min_expected_q

                    # Use Q-values only for the taken actions
                    v1, adv1 = qf1(b_spatial, b_internal)
                    v2, adv2 = qf2(b_spatial, b_internal)

                    adv1_a = adv1.gather(2, b_actions.unsqueeze(2)).squeeze(2)  # [B, n_agents]
                    adv2_a = adv2.gather(2, b_actions.unsqueeze(2)).squeeze(2)  # [B, n_agents]

                    # Construct exact joint Q values
                    qf1_a_values_sum = v1 + adv1_a.sum(dim=1, keepdim=True)  # [B, 1]
                    qf2_a_values_sum = v2 + adv2_a.sum(dim=1, keepdim=True)  # [B, 1]

                qf1_loss = F.mse_loss(qf1_a_values_sum, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values_sum, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR TRAINING
                if use_radio:
                    # Single actor pass returns both factors; single pass per Q-net
                    # returns both factored advantages (raw, mean-centered).
                    (_, log_pi, action_probs,
                     _, radio_log_pi, radio_probs) = actor.get_action_radio(b_spatial, b_internal)
                    with torch.no_grad():
                        v1, adv1_base, radio_adv1_base = qf1(b_spatial, b_internal)
                        v2, adv2_base, radio_adv2_base = qf2(b_spatial, b_internal)

                    adv1 = adv1_base - alpha * log_pi
                    adv1 = adv1 - adv1.mean(dim=2, keepdim=True)
                    adv2 = adv2_base - alpha * log_pi
                    adv2 = adv2 - adv2.mean(dim=2, keepdim=True)
                    r_adv1 = radio_adv1_base - alpha * radio_log_pi
                    r_adv1 = r_adv1 - r_adv1.mean(dim=2, keepdim=True)
                    r_adv2 = radio_adv2_base - alpha * radio_log_pi
                    r_adv2 = r_adv2 - r_adv2.mean(dim=2, keepdim=True)

                    exp_sum1 = v1 + (action_probs * adv1).sum(dim=2).sum(dim=1, keepdim=True) + (radio_probs * r_adv1).sum(dim=2).sum(dim=1, keepdim=True)
                    exp_sum2 = v2 + (action_probs * adv2).sum(dim=2).sum(dim=1, keepdim=True) + (radio_probs * r_adv2).sum(dim=2).sum(dim=1, keepdim=True)
                else:
                    _, log_pi, action_probs = actor.get_action(b_spatial, b_internal)
                    with torch.no_grad():
                        v1, adv1_base = qf1(b_spatial, b_internal)
                        v2, adv2_base = qf2(b_spatial, b_internal)

                    # Incorporate entropy and re-center outside no_grad so gradients flow to actor
                    adv1 = adv1_base - alpha * log_pi
                    adv1 = adv1 - adv1.mean(dim=2, keepdim=True)
                    adv2 = adv2_base - alpha * log_pi
                    adv2 = adv2 - adv2.mean(dim=2, keepdim=True)

                    # Compute expected sum. We minimize the joint expectation.
                    exp_sum1 = v1 + (action_probs * adv1).sum(dim=2).sum(dim=1, keepdim=True)
                    exp_sum2 = v2 + (action_probs * adv2).sum(dim=2).sum(dim=1, keepdim=True)

                min_qf_values_sum = torch.min(exp_sum1, exp_sum2)

                # Because expected values are entirely encapsulated, directly minimize the negative expected Q-value
                actor_loss = -min_qf_values_sum.mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).sum(dim=2).mean()
                    if use_radio:
                        alpha_loss = alpha_loss + (radio_probs.detach() * (-log_alpha.exp() * (radio_log_pi + target_entropy).detach())).sum(dim=2).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                # Update Target Networks
                if update_count % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                update_count += 1
            cached_logical_steps -= args.train_frequency

        # --- Logging steps/sec and updates/sec every 1% of total steps ---
        # Require update_count > 0: the loss/value tensors below are only bound
        # after the first gradient update, so never log before one has run.
        if step_count >= args.total_timesteps//100 and global_step > args.learning_starts and update_count > 0:
            now = time.time()
            elapsed = now - last_log_time
            steps_per_sec = step_count / elapsed
            updates_per_sec = update_count / elapsed if elapsed > 0 else 0.0
            
            writer.add_scalar("charts/SPS", steps_per_sec, global_step)
            writer.add_scalar("losses/qf1_values", qf1_a_values_sum.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values_sum.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            
            print(f"Global Step: {global_step} | SPS: {steps_per_sec:.0f} | Q-Loss: {qf_loss.item() / 2.0:.3f} | Actor-Loss: {actor_loss.item():.3f} left: {(args.total_timesteps-global_step)/steps_per_sec}s r: {sum(episodic_returns[-10:])/10.0}")
            
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            last_log_time = now
            step_count = 0
            update_count = 0

    writer.close()

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
    np.save(
        os.path.join(results_dir, f"sac_{variant}_episodic_returns_run_{args.run_number}.npy"),
        np.array(episodic_returns),
    )