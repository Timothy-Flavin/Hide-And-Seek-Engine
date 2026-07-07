import os
import json
import random
import sys
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Our compiled nets are called at a small fixed set of batch sizes (rollout
# num_envs, update batch_size, plus radio/no-radio paths), so each needs its own
# size-specialized graph. Raise the per-frame recompile cache above the default 8
# so every variant caches instead of falling back to eager ("torch._dynamo hit
# config.cache_size_limit (8)"). Static kernels are kept, so steady-state
# throughput is unaffected; only compile warmup/memory grows.
torch._dynamo.config.cache_size_limit = 32
import tyro

torch.set_float32_matmul_precision("high")

from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv
from RL.MixedObservationEncoder import MixedObservationEncoder, cast_obs
from RL.checkpoint_utils import (
    CheckpointSaver,
    variant_name,
    results_dir_for,
    module_state,
    resume_checkpoint_path,
    load_resume,
    save_resume,
    restore_resume,
    restore_rng,
)
from RL.human_data import (
    load_bc_batcher, load_transition_batcher, assert_demos_match_env,
    load_per_agent_bc_batchers, load_per_agent_transition_batchers,
)
from RL.eval_utils import run_and_log_eval
from RL.benchmark_utils import BenchmarkClock, write_benchmark


@dataclass
class Args:
    exp_name: str = "custom_dqn"
    run_number: int = 1

    # --- Throughput benchmark (see RL/benchmark_utils.py and benchmark.sh) ---
    benchmark: bool = False
    """time ~20s of steady-state throughput and write steps/sec to
    time_files/<machine>.json, then exit (no training or plots)"""
    machine: str = ""
    """machine name for the --benchmark output file (time_files/<machine>.json)"""
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

    # --- Human-BC pre-training loop (all opt-in; defaults reproduce the
    #     existing run_experiments.sh behavior exactly) ---
    human_bc: bool = False
    """add a behavior-cloning loss from recorded human data (decentralized only)"""
    human_cql: bool = False
    """add an OFFLINE-Q + conservative-Q (CQL) loss from recorded human
    transitions (decentralized only). This is the joint online+offline objective
    for the offline-vs-online experiment: online TD stays penalty-free, while the
    offline (human) minibatch gets an offline Bellman backup plus a CQL penalty
    that pushes down out-of-distribution action values. Mutually exclusive with
    --human-bc (CQL wins with a warning if both are set)."""
    cql_coef: float = 1.0
    """weight on the whole offline-Q+CQL term (added to the online TD loss)"""
    cql_alpha: float = 1.0
    """weight on the conservative (logsumexp - taken) penalty within the offline term"""
    bc_coef: float = 0.2
    """weight on the BC cross-entropy term"""
    bc_batch_size: int = 256
    """minibatch size for the BC (or offline-Q/CQL) term"""
    per_agent_nets: bool = False
    """decentralized-only: give each agent index its own independent Q-network
    (encoder+adv+value head) instead of sharing one, so each agent specializes to
    its role. Online VDN and the offline term (CQL or BC) are both routed per-agent
    (each agent's demos back up its own net). Checkpoints use a '_pa' variant suffix
    so they don't collide with the shared-net baseline."""
    resume: bool = False
    """resume optimizer/epsilon/RNG from this run's resume checkpoint, and write
    one at the end (for the 100k-frame-chunk loop)"""
    exploration_timesteps: int = 0
    """epsilon-schedule horizon; 0 -> use total_timesteps (original behavior)"""
    eval_episodes: int = 10
    """episodes to evaluate at step 0 (random) and after training, each clipped to
    the human/training episode length; logs per-agent + team return to
    <level>/<alg>/eval_returns.jsonl (0 disables)"""


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
        # Pinned (page-locked) host memory only pays off for host->device copies,
        pin = torch.device(device).type == "cuda"
        if store_radio:
            self.radio_actions = torch.empty((capacity, n_agents), dtype=torch.int64).contiguous()

        # Spatial obs stored as uint8 (4x smaller pinned buffer); cast to float
        # happens inside the network (cast_obs). Internal vector stays float32.
        self.spatial_obs = torch.empty(
            (capacity, *spatial_shape), dtype=torch.uint8, pin_memory=pin
        ).contiguous()
        self.internal_obs = torch.empty(
            (capacity, *internal_dim), dtype=torch.float32, pin_memory=pin
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
    def __init__(self, spatial_shape, internal_dim, n_agents, centralized=True, num_actions_per_agent=5, use_radio=False, n_radio_actions=0, per_agent=False):
        super().__init__()
        self.n_agents = n_agents
        self.num_actions_per_agent = num_actions_per_agent
        self.spatial_shape = spatial_shape
        self.internal_dim = internal_dim
        self.centralized = centralized
        # Radio ablation: decentralized-only trainable per-agent radio Q-head.
        self.use_radio = use_radio and not centralized
        self.n_radio_actions = n_radio_actions
        # Per-agent nets: decentralized-only, one independent (encoder+adv+v) per
        # agent index so each agent's Q specializes to its role.
        self.per_agent = bool(per_agent and not centralized)

        # Advantage head width = move dims (+ radio dims for the radio ablation).
        self.head_out_dim = num_actions_per_agent + (n_radio_actions if self.use_radio else 0)
        vector_dim = np.prod(internal_dim) if centralized else internal_dim[1]

        def _make_encoder():
            return MixedObservationEncoder(
                spatial_shape=spatial_shape, vector_dim=vector_dim,
                spatial_hidden_dim=128, vector_hidden_dim=32, output_dim=256,
            )

        def _make_adv():
            return nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.head_out_dim))

        def _make_v():
            return nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

        if self.per_agent:
            self.encoders = nn.ModuleList([_make_encoder() for _ in range(n_agents)])
            self.adv_heads = nn.ModuleList([_make_adv() for _ in range(n_agents)])
            self.v_heads = nn.ModuleList([_make_v() for _ in range(n_agents)])
        else:
            self.encoder = _make_encoder()
            self.adv_heads = nn.ModuleList([_make_adv() for _ in range(n_agents if centralized else 1)])
            self.v_head = _make_v()

        if centralized:
            self.get_action = self._get_action_centralized
            self.forward = self._forward_centralized
        elif self.per_agent:
            self.get_action = (self.get_action_radio if self.use_radio
                               else self._get_action_decentralized)
            self.forward = self._forward_decentralized
        elif self.use_radio:
            self.get_action = self.get_action_radio
            self.forward = self._forward_decentralized
        else:
            self.get_action = self._get_action_decentralized
            self.forward = self._forward_decentralized

    # --- Per-agent (non-shared) helpers: each agent index runs through its own
    #     encoder + value/advantage heads. ---
    def _per_agent_v_adv(self, spatial, internal):
        """Return (v_summed[B,1], adv_uncentered[B, n_agents, head_out_dim])."""
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        vs, advs = [], []
        for a in range(self.n_agents):
            x = torch.cat([spatial[:, a].reshape(B, -1), internal[:, a].reshape(B, -1)], dim=-1)
            feats = self.encoders[a](x)
            vs.append(self.v_heads[a](feats))
            advs.append(self.adv_heads[a](feats))
        v = torch.stack(vs, dim=1).sum(dim=1)                     # [B, 1] (VDN sum)
        adv = torch.stack(advs, dim=1)                            # [B, n_agents, head_out_dim]
        return v, adv

    def get_action_radio(self, spatial, internal):
        # Decentralized radio ablation: fused advantage head split into move/radio;
        # take each greedy action. (Argmax is invariant to dueling mean-centering.)
        if self.per_agent:
            _, adv_all = self._per_agent_v_adv(spatial, internal)
        else:
            spatial = cast_obs(spatial)
            B = spatial.shape[0]
            x = torch.cat([spatial.view(B * self.n_agents, -1), internal.view(B * self.n_agents, -1)], dim=-1)
            adv_all = self.adv_heads[0](self.encoder(x)).view(B, self.n_agents, self.head_out_dim)
        move_action = torch.argmax(adv_all[..., :self.num_actions_per_agent], dim=2)
        radio_action = torch.argmax(adv_all[..., self.num_actions_per_agent:], dim=2)
        return move_action, radio_action

    def _get_action_centralized(self, spatial, internal):
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        spatial_flat = spatial.view(B, -1)
        internal_flat = internal.view(B, -1)
        x = torch.cat([spatial_flat, internal_flat], dim=-1)

        feats = self.encoder(x)
        adv_values = torch.stack([head(feats) for head in self.adv_heads], dim=1) # [B, n_agents, num_actions]
        return torch.argmax(adv_values, dim=2)

    def _get_action_decentralized(self, spatial, internal):
        if self.per_agent:
            _, adv_values = self._per_agent_v_adv(spatial, internal)
            return torch.argmax(adv_values[..., :self.num_actions_per_agent], dim=2)
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        spatial_flat = spatial.view(B * self.n_agents, -1)
        internal_flat = internal.view(B * self.n_agents, -1)
        x = torch.cat([spatial_flat, internal_flat], dim=-1)

        feats = self.encoder(x)
        adv_values = self.adv_heads[0](feats)
        adv_values = adv_values.view(B, self.n_agents, self.num_actions_per_agent)
        return torch.argmax(adv_values, dim=2)

    def _forward_centralized(self, spatial, internal):
        spatial = cast_obs(spatial)
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
        if self.per_agent:
            v_value, adv_values = self._per_agent_v_adv(spatial, internal)
        else:
            spatial = cast_obs(spatial)
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

    def bc_logits(self, spatial, internal, agent_idx=0):
        """Per-single-agent BC logits (decentralized): (move, radio).

        Runs the shared encoder + shared advantage head on one agent's ego
        observation (spatial: [B, C, S, S] uint8, internal: [B, D]) and splits
        the output into the move slice and, when ``use_radio``, the radio slice
        (else None). cast_obs handles the uint8->float conversion, exactly as in
        rollouts.
        """
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        x = torch.cat([spatial.view(B, -1), internal.view(B, -1)], dim=-1)
        if self.per_agent:
            out = self.adv_heads[agent_idx](self.encoders[agent_idx](x))
        else:
            out = self.adv_heads[0](self.encoder(x))
        move = out[..., :self.num_actions_per_agent]
        radio = out[..., self.num_actions_per_agent:] if self.use_radio else None
        return move, radio

    def q_single(self, spatial, internal, agent_idx=0):
        """Per-single-agent dueling Q pieces (decentralized): (v, move_adv, radio_adv).

        Runs the encoder + value head + advantage head on ONE agent's ego
        observation (spatial: [B, C, S, S] uint8, internal: [B, D]) and returns
        the value ``v`` [B, 1] plus each advantage factor MEAN-CENTERED over its
        own action dim -- exactly as ``_forward_decentralized`` centers them, so a
        single-agent offline Q ``v + adv_centered[a]`` is consistent with the VDN
        per-agent contribution used online. In per-agent-nets mode ``agent_idx``
        selects that agent's own net. ``radio_adv`` is None without radio.
        """
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        x = torch.cat([spatial.view(B, -1), internal.view(B, -1)], dim=-1)
        if self.per_agent:
            feats = self.encoders[agent_idx](x)
            v = self.v_heads[agent_idx](feats)  # [B, 1]
            out = self.adv_heads[agent_idx](feats)
        else:
            feats = self.encoder(x)
            v = self.v_head(feats)  # [B, 1]
            out = self.adv_heads[0](feats)
        move = out[..., :self.num_actions_per_agent]
        move = move - move.mean(dim=1, keepdim=True)
        if self.use_radio:
            radio = out[..., self.num_actions_per_agent:]
            radio = radio - radio.mean(dim=1, keepdim=True)
        else:
            radio = None
        return v, move, radio


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

    per_agent = bool(args.per_agent_nets and not args.centralized)
    q_network = torch.compile(
        QNetwork(spatial_shape, internal_dim, n_agents, centralized=args.centralized, use_radio=use_radio, n_radio_actions=n_radio_actions, per_agent=per_agent).to(device)
    )
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = torch.compile(
        QNetwork(spatial_shape, internal_dim, n_agents, centralized=args.centralized, use_radio=use_radio, n_radio_actions=n_radio_actions, per_agent=per_agent).to(device)
    )
    target_network.load_state_dict(q_network.state_dict())

    rb = CustomReplayBuffer(
        args.buffer_size, args.num_envs, n_agents, buffer_spatial_shape, internal_dim, device, store_radio=use_radio
    )

    # Result layout: experiments/results/<level>/dqn/ ; weights checkpointed at
    # 20/40/60/80/100% of training under that folder's checkpoints/. Per-agent-nets
    # get a '_pa' suffix; the offline objective gets a '_cql'/'_bc' suffix so RL and
    # RL+offline runs are saved separately and never overwrite each other.
    _obj = ""
    if not args.centralized:
        _obj = "_cql" if args.human_cql else ("_bc" if args.human_bc else "")
    variant = variant_name(args.centralized, args.ego_view, use_radio) + ("_pa" if per_agent else "") + _obj
    results_dir = results_dir_for(args.level, "dqn")
    run_prefix = f"dqn_{variant}_run_{args.run_number}"
    ckpt = CheckpointSaver(results_dir, run_prefix, args.total_timesteps)
    ckpt_state = lambda: module_state(q_network=q_network)

    # --- Resumable-training checkpoint (opt-in via --resume): restore optimizer,
    #     epsilon/cumulative-step and RNG so 100k-frame chunks continue cleanly. ---
    resume_path = resume_checkpoint_path(results_dir, run_prefix)
    cumulative_offset = 0
    if args.resume:
        state = load_resume(resume_path, map_location=device)
        if state is not None:
            restore_resume(state, {"q_network": q_network, "target_network": target_network},
                           optimizers={"optimizer": optimizer})
            restore_rng(state)
            cumulative_offset = int(state.get("cumulative_step", 0))
            print(f"[dqn] Resumed from {resume_path} at {cumulative_offset} cumulative frames.")
        else:
            print(f"[dqn] --resume set but no checkpoint at {resume_path}; starting fresh.")

    # --- Offline human-data terms (opt-in; decentralized only, since human demos
    #     are per-agent ego observations). Two mutually-exclusive objectives:
    #       --human-cql : offline-Q + conservative-Q on full human transitions
    #                     (the online+offline experiment's DQN objective);
    #       --human-bc  : plain behavior-cloning cross-entropy (legacy human loop).
    #     Online TD is never touched by either, so online experiences stay
    #     penalty-free. ---
    use_cql = bool(args.human_cql)
    if args.human_cql and args.human_bc:
        print("[dqn] both --human-cql and --human-bc set; using CQL (offline-Q) and "
              "ignoring --human-bc.")
    bc_batcher = None
    cql_batcher = None
    if use_cql:
        if args.centralized:
            print("[dqn] --human-cql needs a decentralized model (ego human data); "
                  "skipping offline-Q. Re-run with --no-centralized.")
        else:
            cql_batcher = load_transition_batcher(
                args.level, expected_spatial_shape=spatial_shape,
                ego_size=args.ego_size if args.ego_view else None,
                expected_internal_dim=env.agent_internal_dim,
            )
            if cql_batcher is None:
                print(f"[dqn] --human-cql set but no matching human transitions for '{args.level}'; skipping offline-Q.")
            else:
                # Refuse stale demos (goal/entity block unpopulated vs the live env).
                assert_demos_match_env(env, cql_batcher.internal, args.num_envs, n_agents,
                                       device, label="dqn human-cql")
                nets = "per-agent nets" if per_agent else "one shared net"
                print(f"[dqn] offline-Q + CQL enabled with {cql_batcher.n} human transitions "
                      f"(cql_coef={args.cql_coef}, cql_alpha={args.cql_alpha}, {nets}).")
    elif args.human_bc:
        if args.centralized:
            print("[dqn] --human-bc needs a decentralized model (ego human data); "
                  "skipping BC. Re-run with --no-centralized.")
        else:
            bc_batcher = load_bc_batcher(
                args.level, expected_spatial_shape=spatial_shape,
                ego_size=args.ego_size if args.ego_view else None,
                expected_internal_dim=env.agent_internal_dim,
            )
            if bc_batcher is None:
                print(f"[dqn] --human-bc set but no matching human data for '{args.level}'; skipping BC.")
            else:
                # Refuse stale demos (goal/entity block unpopulated vs the live env).
                assert_demos_match_env(env, bc_batcher.internal, args.num_envs, n_agents,
                                       device, label="dqn human-bc")
                nets = "per-agent nets" if per_agent else "one shared net"
                print(f"[dqn] BC enabled with {bc_batcher.n} human frames ({nets}).")

    # Per-agent offline routing: each agent's demos back up its own network.
    cql_batchers_pa = {}
    bc_batchers_pa = {}
    if per_agent and cql_batcher is not None:
        cql_batchers_pa = load_per_agent_transition_batchers(
            args.level, n_agents, expected_spatial_shape=spatial_shape,
            ego_size=args.ego_size if args.ego_view else None,
            expected_internal_dim=env.agent_internal_dim,
        )
        print(f"[dqn] per-agent CQL transition counts by agent index: "
              f"{ {a: b.n for a, b in cql_batchers_pa.items()} }")
    if per_agent and bc_batcher is not None:
        bc_batchers_pa = load_per_agent_bc_batchers(
            args.level, n_agents, expected_spatial_shape=spatial_shape,
            ego_size=args.ego_size if args.ego_view else None,
            expected_internal_dim=env.agent_internal_dim,
        )
        print(f"[dqn] per-agent BC frame counts by agent index: "
              f"{ {a: b.n for a, b in bc_batchers_pa.items()} }")

    # Epsilon-schedule horizon; default reproduces the original math exactly.
    epsilon_horizon = args.exploration_timesteps or args.total_timesteps
    epsilon = args.start_e

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
    cql_losses = []  # (offline_td, cql_gap) per update when --human-cql is active

    # --- Post-chunk evaluation (per-agent + team return over args.eval_episodes
    #     episodes, each clipped to the human/training episode length). Also run
    #     once at step 0 (first chunk) for the untrained/random baseline. ---
    eval_max_steps = int(getattr(env, "max_frames", 250))
    with open(os.path.join(args.level, "agents.json")) as _f:
        eval_agent_names = list(json.load(_f).keys())

    def _eval_act(spatial, internal):
        with torch.no_grad():
            if use_radio:
                move_action, radio_action = q_network.get_action(spatial, internal)
                radio = radio_action.cpu().numpy().astype(np.int32)
            else:
                move_action = q_network.get_action(spatial, internal)
                radio = np.zeros((args.num_envs, n_agents), dtype=np.int32)
        return ACTION_MAP[move_action.cpu().numpy()], radio

    def run_eval(cumulative_step):
        if args.eval_episodes <= 0:
            return
        run_and_log_eval(env, _eval_act, args.num_envs, n_agents,
                         level=args.level, alg="dqn", prefix=f"dqn_{variant}_run_{args.run_number}",
                         agent_names=eval_agent_names, cumulative_step=cumulative_step,
                         n_episodes=args.eval_episodes, max_ep_steps=eval_max_steps)

    if not args.benchmark and cumulative_offset == 0:
        run_eval(0)  # random-policy baseline for the no-human step-0 point

    obs = env._get_obs_dict()
    spatial = obs["spatial"]
    internal = obs["internal"]

    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)

    # Benchmark mode: loop until the steady-state clock says stop (see below).
    bench = BenchmarkClock() if args.benchmark else None
    n_iterations = args.total_timesteps // args.num_envs
    if args.benchmark:
        n_iterations = 10 ** 9

    for iteration in range(n_iterations):
        global_step = iteration * args.num_envs
        updated = False  # did a gradient update run this iteration? (benchmark warmup)

        # cumulative_offset is 0 unless --resume, so this reproduces the original
        # schedule exactly for run_experiments.sh; when resuming, the 100k-frame
        # chunks continue decaying along a fixed horizon.
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * epsilon_horizon,
            cumulative_offset + global_step,
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

                if bc_batcher is not None:
                    if per_agent and bc_batchers_pa:
                        bc_loss = 0.0
                        for a, batcher in bc_batchers_pa.items():
                            s_a, i_a, act_a, rad_a = batcher.sample(args.bc_batch_size, device)
                            ml_a, rl_a = q_network.bc_logits(s_a, i_a, agent_idx=a)
                            la = F.cross_entropy(ml_a, act_a)
                            if rl_a is not None and rad_a is not None:
                                la = la + F.cross_entropy(rl_a, rad_a)
                            bc_loss = bc_loss + la
                    else:
                        bc_spatial, bc_internal, bc_actions, bc_radio = bc_batcher.sample(args.bc_batch_size, device)
                        bc_move_logits, bc_radio_logits = q_network.bc_logits(bc_spatial, bc_internal)
                        bc_loss = F.cross_entropy(bc_move_logits, bc_actions)
                        # Clone the human's radio action too, so the radio head learns
                        # to broadcast from demos (not RL alone).
                        if bc_radio_logits is not None and bc_radio is not None:
                            bc_loss = bc_loss + F.cross_entropy(bc_radio_logits, bc_radio)
                    loss = loss + args.bc_coef * bc_loss

                if cql_batcher is not None:
                    # Offline-Q + conservative-Q on human transitions. Single-agent
                    # ego backup: Q(s,a) = v + adv_centered[a] (VDN per-agent piece).
                    # The team reward matches the online summed-reward target, so
                    # offline and online backups are comparable. The online TD loss
                    # above is left untouched -> ONLINE experiences never see the
                    # conservative penalty (only the offline minibatch does). Under
                    # per-agent nets each agent's transitions back up its own net.
                    pairs = (list(cql_batchers_pa.items()) if (per_agent and cql_batchers_pa)
                             else [(0, cql_batcher)])
                    offline_td_sum = 0.0
                    cql_gap_sum = 0.0
                    for aidx, batcher in pairs:
                        (o_s, o_i, o_ns, o_ni, o_move, o_radio,
                         o_r, o_d) = batcher.sample(args.bc_batch_size, device)
                        with torch.no_grad():
                            tv, tmove, tradio = target_network.q_single(o_ns, o_ni, agent_idx=aidx)
                            q_next = tv + tmove.max(dim=1, keepdim=True).values
                            if tradio is not None:
                                q_next = q_next + tradio.max(dim=1, keepdim=True).values
                            cql_target = o_r + args.gamma * (1 - o_d) * q_next

                        v_o, move_o, radio_o = q_network.q_single(o_s, o_i, agent_idx=aidx)
                        move_taken = move_o.gather(1, o_move.unsqueeze(1))          # [B, 1]
                        q_pred = v_o + move_taken
                        # CQL(H) gap: push down logsumexp_a Q(s,a) toward Q(s,a_data).
                        # The value head cancels, so it is a sum of per-factor gaps.
                        cql_gap = torch.logsumexp(move_o, dim=1, keepdim=True) - move_taken
                        if radio_o is not None and o_radio is not None:
                            radio_taken = radio_o.gather(1, o_radio.unsqueeze(1))    # [B, 1]
                            q_pred = q_pred + radio_taken
                            cql_gap = cql_gap + (torch.logsumexp(radio_o, dim=1, keepdim=True) - radio_taken)

                        offline_td_sum = offline_td_sum + F.mse_loss(q_pred, cql_target)
                        cql_gap_sum = cql_gap_sum + cql_gap.mean()

                    offline_loss = offline_td_sum + args.cql_alpha * cql_gap_sum
                    loss = loss + args.cql_coef * offline_loss
                    cql_losses.append((float(offline_td_sum), float(cql_gap_sum)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                q_losses.append(loss.item())
                update_count += 1
                updated = True
            cached_logical_steps -= args.train_frequency

        # --- Logging steps/sec and updates/sec every 1000 env steps ---
        if step_count >= args.total_timesteps//100:
            now = time.time()
            elapsed = now - last_log_time
            steps_per_sec = step_count / elapsed
            updates_per_sec = update_count / elapsed if elapsed > 0 else 0.0
            cql_str = ""
            if cql_losses:
                recent = cql_losses[-100:]
                td_m = sum(t for t, _ in recent) / len(recent)
                gap_m = sum(g for _, g in recent) / len(recent)
                cql_str = f" offline_td: {td_m:.3f} cql_gap: {gap_m:.3f}"
            print(
                f"Steps/sec: {steps_per_sec:.2f}, up/sec: {updates_per_sec:.2f} qloss: {sum(q_losses[-100:])/100}, return: {sum(episodic_returns[-100:])/100}{cql_str} left: {(args.total_timesteps-global_step)/steps_per_sec}s"
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

        # Periodic weight checkpoints (20/40/60/80/100% of training).
        ckpt.maybe_save(global_step, ckpt_state)

        if bench is not None and bench.tick(args.num_envs, updated):
            break

    if args.benchmark:
        summary = bench.summary(args.num_envs)
        path, key = write_benchmark(args.machine, args.level, "dqn", variant, summary)
        print(f"[benchmark] {key}: {summary['steps_per_sec']:.1f} steps/s "
              f"({summary['steps']} steps / {summary['seconds']:.1f}s) -> {path}")
        sys.exit(0)

    ckpt.flush_remaining(ckpt_state)  # guarantee the 100% checkpoint exists

    # Resume checkpoint for the next 100k-frame chunk (opt-in via --resume).
    if args.resume:
        save_resume(
            resume_path,
            models={"q_network": q_network, "target_network": target_network},
            optimizers={"optimizer": optimizer},
            cumulative_step=cumulative_offset + args.total_timesteps,
            extra={"epsilon": float(epsilon), "epsilon_horizon": int(epsilon_horizon)},
        )
        print(f"[dqn] Saved resume checkpoint -> {resume_path} "
              f"({cumulative_offset + args.total_timesteps} cumulative frames).")

    # Post-training evaluation: log per-agent/team return for this chunk.
    run_eval(cumulative_offset + args.total_timesteps)

    # Plot episodic returns at the end (results in experiments/results/<level>/dqn/)
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
