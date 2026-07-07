import os
import json
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Our compiled net is called at a small fixed set of batch sizes (rollout
# num_envs, update minibatch, plus radio/no-radio paths), so each needs its own
# size-specialized graph. Raise the per-frame recompile cache above the default 8
# so every variant caches instead of falling back to eager ("torch._dynamo hit
# config.cache_size_limit (8)"). Static kernels are kept, so steady-state
# throughput is unaffected; only compile warmup/memory grows.
torch._dynamo.config.cache_size_limit = 32
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

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
from RL.human_data import load_bc_batcher, load_per_agent_bc_batchers, assert_demos_match_env
from RL.eval_utils import run_and_log_eval
from RL.benchmark_utils import BenchmarkClock, write_benchmark
import torch.nn.functional as F


@dataclass
class Args:
    exp_name: str = "custom_ppo"
    torch_threads: int = 0
    """the name of this experiment"""

    # --- Throughput benchmark (see RL/benchmark_utils.py and benchmark.sh) ---
    benchmark: bool = False
    """time ~20s of steady-state throughput and write steps/sec to
    time_files/<machine>.json, then exit (no training or plots)"""
    machine: str = ""
    """machine name for the --benchmark output file (time_files/<machine>.json)"""
    centralized: bool = True
    """whether to use centralized or individual PPO"""
    ego_view: bool = False
    """ego-centric obs: fixed window centered on each agent (use with --no-centralized)"""
    ego_size: int = 32
    """side length of the ego-centric obs window when ego_view is set"""
    level: str = "levels/test_level"
    """path to the level directory (expects level.png, tiles.json, agents.json, survivors.json)"""
    use_radio: bool = False
    """decentralized-only ablation: add a trainable per-agent radio head so agents learn to share observations (use with --no-centralized)"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity = None
    """the entity (team) of wandb's project"""
    run_number: int = 1
    """the run number for this experiment"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # --- Human-BC pre-training loop (all opt-in; defaults reproduce the
    #     existing run_experiments.sh behavior exactly) ---
    human_bc: bool = False
    """add a behavior-cloning loss from recorded human data (decentralized only)"""
    bc_coef: float = 1.0
    """weight on the BC cross-entropy term"""
    bc_batch_size: int = 256
    """minibatch size for the BC term"""
    per_agent_nets: bool = False
    """decentralized-only: give each agent index its own independent network
    (encoder+head+critic) instead of sharing one across agents, so each agent
    specializes to its role. Online RL and the BC term are both routed per-agent
    (each agent's demos train its own net). Checkpoints use a '_pa' variant suffix
    so they don't collide with the shared-net baseline."""
    bc_separate: bool = False
    """train the BC term on a SEPARATE policy network (its own optimizer) instead
    of adding it to the online policy's loss. Diagnostic: isolates whether a shared
    head (non-identifiability of the online vs human action for the same obs) is
    what blocks learning. The online net then trains on PURE RL (no BC gradient)
    and is what gets evaluated for team score, while the human demos train an
    independent net whose imitation accuracy is logged (charts/bc_move_acc)."""
    resume: bool = False
    """resume weights/optimizer/RNG from this run's resume checkpoint, and write
    one at the end (for the 100k-frame-chunk loop)"""
    eval_episodes: int = 10
    """episodes to evaluate at step 0 (random) and after training, each clipped to
    the human/training episode length; logs per-agent + team return to
    <level>/<alg>/eval_returns.jsonl (0 disables)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, spatial_shape, internal_dim, n_agents, num_actions_per_agent=5, centralized=True, use_radio=False, n_radio_actions=0, per_agent=False):
        super().__init__()
        self.n_agents = n_agents
        self.num_actions_per_agent = num_actions_per_agent
        self.centralized = centralized
        # Radio ablation: decentralized-only trainable per-agent radio head.
        self.use_radio = use_radio and not centralized
        self.n_radio_actions = n_radio_actions
        # Per-agent nets: decentralized-only, one independent (encoder+head+critic)
        # per agent index so each agent specializes to its own role instead of
        # sharing one set of weights across all agents.
        self.per_agent = bool(per_agent and not centralized)

        self.head_out_dim = num_actions_per_agent + (n_radio_actions if self.use_radio else 0)
        vector_dim = np.prod(internal_dim) if centralized else internal_dim[1]

        def _make_encoder():
            return MixedObservationEncoder(
                spatial_shape=spatial_shape, vector_dim=vector_dim,
                spatial_hidden_dim=128, vector_hidden_dim=32, output_dim=256,
            )

        def _make_head():
            return nn.Sequential(
                layer_init(nn.Linear(256, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, self.head_out_dim), std=0.01),
            )

        def _make_critic():
            return nn.Sequential(
                layer_init(nn.Linear(256, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 1), std=1.0),
            )

        if self.per_agent:
            # One full network per agent index.
            self.encoders = nn.ModuleList([_make_encoder() for _ in range(n_agents)])
            self.actor_heads = nn.ModuleList([_make_head() for _ in range(n_agents)])
            self.critics = nn.ModuleList([_make_critic() for _ in range(n_agents)])
        else:
            # Shared encoder/critic; one head (decentralized) or n_agents (centralized).
            self.encoder = _make_encoder()
            self.actor_heads = nn.ModuleList([_make_head() for _ in range(n_agents if centralized else 1)])
            self.critic = _make_critic()

        if centralized:
            self.get_value = self._get_value_centralized
            self.get_action_and_value = self._get_action_and_value_centralized
        elif self.per_agent:
            self.get_value = self._get_value_decentralized_pa
            self.get_action_and_value = (
                self._get_action_and_value_radio_pa if self.use_radio
                else self._get_action_and_value_decentralized_pa
            )
        elif self.use_radio:
            self.get_value = self._get_value_decentralized
            self.get_action_and_value = self._get_action_and_value_radio
        else:
            self.get_value = self._get_value_decentralized
            self.get_action_and_value = self._get_action_and_value_decentralized

    # --- Per-agent (non-shared) decentralized forwards. Each agent index runs
    #     through its own encoder+head+critic; outputs are stacked to the same
    #     [B, n_agents, ...] layout the shared path returns. ---
    def _per_agent_forward(self, spatial, internal):
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        moves, radios, vals = [], [], []
        for a in range(self.n_agents):
            x = torch.cat([spatial[:, a].reshape(B, -1), internal[:, a].reshape(B, -1)], dim=-1)
            feats = self.encoders[a](x)
            logits = self.actor_heads[a](feats)
            moves.append(logits[..., :self.num_actions_per_agent])
            if self.use_radio:
                radios.append(logits[..., self.num_actions_per_agent:])
            vals.append(self.critics[a](feats))
        move_logits = torch.stack(moves, dim=1)                       # [B, n, A]
        radio_logits = torch.stack(radios, dim=1) if self.use_radio else None
        value = torch.cat(vals, dim=1)                                 # [B, n]
        return move_logits, radio_logits, value

    def _get_action_and_value_decentralized_pa(self, spatial, internal, action=None):
        move_logits, _, value = self._per_agent_forward(spatial, internal)
        probs = Categorical(logits=move_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

    def _get_action_and_value_radio_pa(self, spatial, internal, action=None, radio_action=None):
        move_logits, radio_logits, value = self._per_agent_forward(spatial, internal)
        move_probs = Categorical(logits=move_logits)
        radio_probs = Categorical(logits=radio_logits)
        if action is None:
            action = move_probs.sample()
        if radio_action is None:
            radio_action = radio_probs.sample()
        return (
            action, move_probs.log_prob(action), move_probs.entropy(), value,
            radio_action, radio_probs.log_prob(radio_action), radio_probs.entropy(),
        )

    def _get_value_decentralized_pa(self, spatial, internal):
        _, _, value = self._per_agent_forward(spatial, internal)
        return value

    def _get_action_and_value_radio(self, spatial, internal, action=None, radio_action=None):
        spatial = cast_obs(spatial)
        # Decentralized radio ablation: ONE encoder pass, ONE fused actor head of
        # width (move + radio); split the logits (views) into the two factors.
        B = spatial.shape[0]
        x = torch.cat([spatial.view(B * self.n_agents, -1), internal.view(B * self.n_agents, -1)], dim=-1)
        feats = self.encoder(x)

        logits_all = self.actor_heads[0](feats).view(B, self.n_agents, self.head_out_dim)
        move_logits = logits_all[..., :self.num_actions_per_agent]
        radio_logits = logits_all[..., self.num_actions_per_agent:]

        move_probs = Categorical(logits=move_logits)
        radio_probs = Categorical(logits=radio_logits)
        if action is None:
            action = move_probs.sample()
        if radio_action is None:
            radio_action = radio_probs.sample()

        value = self.critic(feats).view(B, self.n_agents)
        return (
            action, move_probs.log_prob(action), move_probs.entropy(), value,
            radio_action, radio_probs.log_prob(radio_action), radio_probs.entropy(),
        )

    def _get_value_centralized(self, spatial, internal):
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        x = torch.cat([spatial.view(B, -1), internal.view(B, -1)], dim=-1)
        feats = self.encoder(x)
        return self.critic(feats)

    def _get_value_decentralized(self, spatial, internal):
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        x = torch.cat([spatial.view(B * self.n_agents, -1), internal.view(B * self.n_agents, -1)], dim=-1)
        feats = self.encoder(x)
        return self.critic(feats).view(B, self.n_agents)

    def _get_action_and_value_centralized(self, spatial, internal, action=None):
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        x = torch.cat([spatial.view(B, -1), internal.view(B, -1)], dim=-1)
        feats = self.encoder(x)
        
        # Calculate Logits
        logits = torch.stack([head(feats) for head in self.actor_heads], dim=1) # [B, n_agents, num_actions]
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample() # [B, n_agents]
            
        logprob = probs.log_prob(action) # [B, n_agents]
        entropy = probs.entropy() # [B, n_agents]
        
        return action, logprob, entropy, self.critic(feats).expand(B, self.n_agents)

    def _get_action_and_value_decentralized(self, spatial, internal, action=None):
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        x = torch.cat([spatial.view(B * self.n_agents, -1), internal.view(B * self.n_agents, -1)], dim=-1)
        feats = self.encoder(x)
        
        logits = self.actor_heads[0](feats).view(B, self.n_agents, self.num_actions_per_agent)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample() # [B, n_agents]
            
        logprob = probs.log_prob(action) # [B, n_agents]
        entropy = probs.entropy() # [B, n_agents]

        return action, logprob, entropy, self.critic(feats).view(B, self.n_agents)

    def bc_logits(self, spatial, internal, agent_idx=0):
        """Per-single-agent BC logits (decentralized): (move, radio).

        Runs one agent's ego obs (spatial: [B, C, S, S] uint8, internal: [B, D])
        through its policy and returns the move slice and, when ``use_radio``, the
        radio slice (else None). In per-agent-nets mode ``agent_idx`` selects that
        agent's own network; otherwise the shared encoder/head is used.
        """
        spatial = cast_obs(spatial)
        B = spatial.shape[0]
        x = torch.cat([spatial.view(B, -1), internal.view(B, -1)], dim=-1)
        if self.per_agent:
            feats = self.encoders[agent_idx](x)
            out = self.actor_heads[agent_idx](feats)
        else:
            feats = self.encoder(x)
            out = self.actor_heads[0](feats)
        move = out[..., :self.num_actions_per_agent]
        radio = out[..., self.num_actions_per_agent:] if self.use_radio else None
        return move, radio


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
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.benchmark:
        # Loop until the steady-state benchmark clock says stop (see below).
        args.num_iterations = 10 ** 9
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
    # Single-agent spatial map shape (C, H, W) or (C, ego, ego); robust to ego mode.
    spatial_shape = env.map_spatial_shape
    internal_dim = (env.config.n_agents, env.agent_internal_dim)
    num_actions_per_agent = 5

    if args.use_radio and args.centralized:
        raise ValueError("--use-radio is a decentralized ablation; run it with --no-centralized")
    # Radio action space: 0 = no broadcast, 1..n_agents-1 = share with a peer.
    n_radio_actions = n_agents
    use_radio = bool(args.use_radio and not args.centralized)

    per_agent = bool(args.per_agent_nets and not args.centralized)
    agent = torch.compile(Agent(spatial_shape, internal_dim, n_agents, num_actions_per_agent, args.centralized, use_radio, n_radio_actions, per_agent).to(device))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Separate BC network (diagnostic; --bc-separate). Trains ONLY on the human
    #     demos with its own optimizer, so the online `agent` above gets zero BC
    #     gradient. Created here (before resume restore) so it participates in the
    #     resume state. Only meaningful for the decentralized BC path. ---
    bc_separate = bool(args.human_bc and args.bc_separate and not args.centralized)
    bc_agent = None
    bc_optimizer = None
    if bc_separate:
        bc_agent = torch.compile(Agent(spatial_shape, internal_dim, n_agents, num_actions_per_agent, args.centralized, use_radio, n_radio_actions, per_agent).to(device))
        bc_optimizer = optim.Adam(bc_agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Result layout: experiments/results/<level>/ppo/ ; weights checkpointed at
    # 20/40/60/80/100% of training under that folder's checkpoints/. Per-agent-nets
    # get a '_pa' suffix; BC that actually shapes the online policy gets a '_bc'
    # suffix so RL and RL+BC runs save separately (apples-to-apples). --bc-separate
    # is NOT tagged: its saved/evaluated policy is pure RL (the BC term only trains
    # the throwaway net), so it belongs with the plain RL models.
    _obj = "_bc" if (args.human_bc and not args.centralized and not bc_separate) else ""
    variant = variant_name(args.centralized, args.ego_view, use_radio) + ("_pa" if per_agent else "") + _obj
    results_dir = results_dir_for(args.level, "ppo")
    run_prefix = f"ppo_{variant}_run_{args.run_number}"
    ckpt = CheckpointSaver(results_dir, run_prefix, args.total_timesteps)
    ckpt_state = lambda: module_state(agent=agent)

    # --- Resumable-training checkpoint (opt-in via --resume). ---
    resume_path = resume_checkpoint_path(results_dir, run_prefix)
    cumulative_offset = 0
    if args.resume:
        state = load_resume(resume_path, map_location=device)
        if state is not None:
            resume_models = {"agent": agent}
            resume_opts = {"optimizer": optimizer}
            if bc_separate:
                resume_models["bc_agent"] = bc_agent
                resume_opts["bc_optimizer"] = bc_optimizer
            restore_resume(state, resume_models, optimizers=resume_opts)
            restore_rng(state)
            cumulative_offset = int(state.get("cumulative_step", 0))
            print(f"[ppo] Resumed from {resume_path} at {cumulative_offset} cumulative frames.")
        else:
            print(f"[ppo] --resume set but no checkpoint at {resume_path}; starting fresh.")

    # --- Behavior-cloning term (opt-in via --human-bc; decentralized only). ---
    bc_batcher = None
    if args.human_bc:
        if args.centralized:
            print("[ppo] --human-bc needs a decentralized model (ego human data); "
                  "skipping BC. Re-run with --no-centralized.")
        else:
            bc_batcher = load_bc_batcher(
                args.level, expected_spatial_shape=spatial_shape,
                ego_size=args.ego_size if args.ego_view else None,
                expected_internal_dim=env.agent_internal_dim,
            )
            if bc_batcher is None:
                print(f"[ppo] --human-bc set but no matching human data for '{args.level}'; skipping BC.")
            else:
                # Refuse stale demos (goal/entity block unpopulated vs the live env).
                assert_demos_match_env(env, bc_batcher.internal, args.num_envs, n_agents,
                                       device, label="ppo human-bc")
                mode = ("SEPARATE net (online policy trains pure-RL)" if bc_separate
                        else "shared net")
                nets = "per-agent nets" if per_agent else "one shared net"
                print(f"[ppo] BC enabled with {bc_batcher.n} human frames "
                      f"(coef={args.bc_coef}, {mode}, {nets}).")

    # Per-agent BC routing: each agent's demos train its own network.
    bc_batchers_pa = {}
    if per_agent and bc_batcher is not None:
        bc_batchers_pa = load_per_agent_bc_batchers(
            args.level, n_agents, expected_spatial_shape=spatial_shape,
            ego_size=args.ego_size if args.ego_view else None,
            expected_internal_dim=env.agent_internal_dim,
        )
        counts = {a: b.n for a, b in bc_batchers_pa.items()}
        print(f"[ppo] per-agent BC frame counts by agent index: {counts}")

    # ALGO Logic: Storage setup. The spatial rollout is the dominant on-GPU
    # tensor (num_steps x num_envs x C x H x W); storing it as uint8 cuts its
    # VRAM 4x. The network casts to float on read (cast_obs). obs_internal stays
    # float32.
    if args.centralized:
        obs_spatial = torch.empty((args.num_steps, args.num_envs) + spatial_shape, dtype=torch.uint8).to(device)
        obs_internal = torch.empty((args.num_steps, args.num_envs) + internal_dim).to(device)
    else:
        obs_spatial = torch.empty((args.num_steps, args.num_envs, n_agents) + spatial_shape, dtype=torch.uint8).to(device)
        obs_internal = torch.empty((args.num_steps, args.num_envs, n_agents, env.agent_internal_dim)).to(device)
    
    actions = torch.empty((args.num_steps, args.num_envs, n_agents)).to(device)
    logprobs = torch.empty((args.num_steps, args.num_envs, n_agents)).to(device)
    rewards = torch.empty((args.num_steps, args.num_envs, n_agents)).to(device)
    dones = torch.empty((args.num_steps, args.num_envs, n_agents)).to(device)
    values = torch.empty((args.num_steps, args.num_envs, n_agents)).to(device)
    # Radio-ablation storage: joint (move+radio) logprobs go in `logprobs`; the
    # radio action index is stored separately so it can be re-evaluated.
    if use_radio:
        radio_actions_buf = torch.empty((args.num_steps, args.num_envs, n_agents), dtype=torch.long).to(device)

    # --- Post-chunk evaluation (per-agent + team return over args.eval_episodes
    #     episodes, each clipped to the human/training episode length so
    #     human-in-the-loop and policy-only numbers are apples-to-apples). Also
    #     run once at step 0 (first chunk) for the untrained/random baseline. ---
    eval_max_steps = int(getattr(env, "max_frames", 250))
    with open(os.path.join(args.level, "agents.json")) as _f:
        eval_agent_names = list(json.load(_f).keys())

    def _eval_act(spatial, internal):
        with torch.no_grad():
            if use_radio:
                action, _, _, _, radio_action, _, _ = agent.get_action_and_value(spatial, internal)
                radio = radio_action.cpu().numpy().astype(np.int32)
            else:
                action, _, _, _ = agent.get_action_and_value(spatial, internal)
                radio = np.zeros((args.num_envs, n_agents), dtype=np.int32)
        return ACTION_MAP[action.cpu().numpy()], radio

    def run_eval(cumulative_step):
        if args.eval_episodes <= 0:
            return
        run_and_log_eval(env, _eval_act, args.num_envs, n_agents,
                         level=args.level, alg="ppo", prefix=f"ppo_{variant}_run_{args.run_number}",
                         agent_names=eval_agent_names, cumulative_step=cumulative_step,
                         n_episodes=args.eval_episodes, max_ep_steps=eval_max_steps)

    if not args.benchmark and cumulative_offset == 0:
        run_eval(0)  # random-policy baseline for the no-human step-0 point

    # Start the game
    global_step = 0
    start_time = time.time()

    obs = env._get_obs_dict()
    next_spatial = obs["spatial"]
    next_internal = obs["internal"]
    next_done = torch.empty(args.num_envs).to(device)

    # Manual episodic trackers for logging
    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    episodic_returns = []

    bench = BenchmarkClock() if args.benchmark else None

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            
            obs_spatial[step] = next_spatial
            obs_internal[step] = next_internal
            dones[step] = next_done.unsqueeze(1).expand(-1, n_agents)

            with torch.no_grad():
                if use_radio:
                    # Single forward pass yields both factored actions.
                    action, logprob, _, value, radio_action, radio_logprob, _ = agent.get_action_and_value(next_spatial, next_internal)
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_spatial, next_internal)
                values[step] = value
            actions[step] = action

            # Map Discrete to Continuous XY using ACTION_MAP
            actions_np = action.cpu().numpy()
            move_actions = ACTION_MAP[actions_np] # Shape: (num_envs, n_agents, 2)

            if use_radio:
                radio_actions_buf[step] = radio_action
                logprobs[step] = logprob + radio_logprob  # joint logprob for PPO ratio
                radio_actions = radio_action.cpu().numpy().astype(np.int32)
            else:
                logprobs[step] = logprob
                radio_actions = np.zeros((args.num_envs, n_agents), dtype=np.int32)

            next_obs, rewards_raw, terminations, truncations, infos = env.step(move_actions, radio_actions)

            # Sum rewards over all agents for centralized evaluation
            env_rewards = rewards_raw.sum(dim=1) 
            if args.centralized:
                rewards[step] = env_rewards.unsqueeze(1).expand(-1, n_agents)
            else:
                rewards[step] = rewards_raw
            
            episode_rewards += env_rewards.cpu().numpy()
            episode_lengths += 1

            if terminations.any() or truncations.any():
                for e in range(args.num_envs):
                    if terminations[e] or truncations[e]:
                        writer.add_scalar("charts/episodic_return", episode_rewards[e], global_step)
                        writer.add_scalar("charts/episodic_length", episode_lengths[e], global_step)
                        episodic_returns.append(episode_rewards[e])
                        episode_rewards[e] = 0.0
                        episode_lengths[e] = 0
                        env.reset_env(e) # Manual gridworld reset
                
                # Fetch fresh observations reflecting resets across the batch
                obs_dict = env._get_obs_dict()
                next_spatial = obs_dict["spatial"]
                next_internal = obs_dict["internal"]
            else:
                next_spatial = next_obs["spatial"]
                next_internal = next_obs["internal"]
                
            next_done = torch.logical_or(terminations, truncations).float()
            dones[step] = next_done.unsqueeze(1).expand(-1, n_agents)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_spatial, next_internal)
            if args.centralized:
                next_value = next_value.expand(-1, n_agents)
            
            advantages = torch.empty_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.unsqueeze(1).expand(-1, n_agents)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        if args.centralized:
            b_obs_spatial = obs_spatial.reshape((-1,) + spatial_shape)
            b_obs_internal = obs_internal.reshape((-1,) + internal_dim)
        else:
            b_obs_spatial = obs_spatial.reshape((-1, n_agents) + spatial_shape)
            b_obs_internal = obs_internal.reshape((-1, n_agents, env.agent_internal_dim))
        
        b_logprobs = logprobs.reshape((-1, n_agents))
        b_actions = actions.reshape((-1, n_agents))
        b_advantages = advantages.reshape((-1, n_agents))
        b_returns = returns.reshape((-1, n_agents))
        b_values = values.reshape((-1, n_agents))
        if use_radio:
            b_radio_actions = radio_actions_buf.reshape((-1, n_agents))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if use_radio:
                    # Single forward pass returns both factors; joint (move+radio)
                    # logprob/entropy so the PPO ratio and entropy bonus cover both.
                    (_, move_newlogprob, move_entropy, newvalue,
                     _, radio_newlogprob, radio_entropy) = agent.get_action_and_value(
                        b_obs_spatial[mb_inds],
                        b_obs_internal[mb_inds],
                        b_actions.long()[mb_inds],
                        b_radio_actions[mb_inds],
                    )
                    newlogprob = move_newlogprob + radio_newlogprob
                    entropy = move_entropy + radio_entropy
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs_spatial[mb_inds],
                        b_obs_internal[mb_inds],
                        b_actions.long()[mb_inds]
                    )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.reshape(-1)
                mb_returns = b_returns[mb_inds].reshape(-1)
                mb_values = b_values[mb_inds].reshape(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                if bc_batcher is not None:
                    # In separate mode the BC term trains its own net; the online
                    # `agent` is left untouched (pure RL). Otherwise it's added to
                    # the shared online loss as before.
                    bc_net = bc_agent if bc_separate else agent
                    if per_agent and bc_batchers_pa:
                        # Route each agent's demos through its own network.
                        bc_loss = 0.0
                        for a, batcher in bc_batchers_pa.items():
                            s_a, i_a, act_a, rad_a = batcher.sample(args.bc_batch_size, device)
                            ml_a, rl_a = bc_net.bc_logits(s_a, i_a, agent_idx=a)
                            la = F.cross_entropy(ml_a, act_a)
                            if rl_a is not None and rad_a is not None:
                                la = la + F.cross_entropy(rl_a, rad_a)
                            bc_loss = bc_loss + la
                    else:
                        bc_spatial, bc_internal, bc_actions, bc_radio = bc_batcher.sample(args.bc_batch_size, device)
                        bc_move_logits, bc_radio_logits = bc_net.bc_logits(bc_spatial, bc_internal)
                        bc_loss = F.cross_entropy(bc_move_logits, bc_actions)
                        # Clone the human's radio action too, so the radio head learns
                        # to broadcast from demos (not RL alone).
                        if bc_radio_logits is not None and bc_radio is not None:
                            bc_loss = bc_loss + F.cross_entropy(bc_radio_logits, bc_radio)
                    if bc_separate:
                        bc_optimizer.zero_grad()
                        (args.bc_coef * bc_loss).backward()
                        nn.utils.clip_grad_norm_(bc_agent.parameters(), args.max_grad_norm)
                        bc_optimizer.step()
                    else:
                        loss = loss + args.bc_coef * bc_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # BC imitation accuracy on a held-in demo sample -- how well the BC-trained
        # net reproduces the human's move on demo observations. Per-agent-nets logs
        # both the overall accuracy and each agent's own net (charts/bc_move_acc_a).
        if bc_batcher is not None:
            bc_net = bc_agent if bc_separate else agent
            with torch.no_grad():
                if per_agent and bc_batchers_pa:
                    correct = total = 0
                    for a, batcher in bc_batchers_pa.items():
                        s_a, i_a, act_a, _ = batcher.sample(2048, device)
                        ml_a, _ = bc_net.bc_logits(s_a, i_a, agent_idx=a)
                        c = (ml_a.argmax(-1) == act_a).float().sum().item()
                        correct += c; total += act_a.shape[0]
                        writer.add_scalar(f"charts/bc_move_acc_{a}", c / act_a.shape[0], global_step)
                    bc_acc = correct / max(total, 1)
                else:
                    s_bc, i_bc, a_bc, _ = bc_batcher.sample(2048, device)
                    bc_move_logits, _ = bc_net.bc_logits(s_bc, i_bc)
                    bc_acc = (bc_move_logits.argmax(-1) == a_bc).float().mean().item()
            writer.add_scalar("charts/bc_move_acc", bc_acc, global_step)
        
        sps = int(global_step / (time.time() - start_time))
        print(f"global_step={global_step}, SPS={sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

        # Periodic weight checkpoints (20/40/60/80/100% of training).
        ckpt.maybe_save(global_step, ckpt_state)

        # PPO runs one full update per iteration, so every iteration is
        # update-containing; count env-frames as the whole rollout batch.
        if bench is not None and bench.tick(args.batch_size, True):
            break

    if args.benchmark:
        summary = bench.summary(args.num_envs)
        path, key = write_benchmark(args.machine, args.level, "ppo", variant, summary)
        print(f"[benchmark] {key}: {summary['steps_per_sec']:.1f} steps/s "
              f"({summary['steps']} steps / {summary['seconds']:.1f}s) -> {path}")
        writer.close()
        sys.exit(0)

    writer.close()
    ckpt.flush_remaining(ckpt_state)  # guarantee the 100% checkpoint exists

    # Resume checkpoint for the next 100k-frame chunk (opt-in via --resume).
    if args.resume:
        save_models = {"agent": agent}
        save_opts = {"optimizer": optimizer}
        if bc_separate:
            save_models["bc_agent"] = bc_agent
            save_opts["bc_optimizer"] = bc_optimizer
        save_resume(
            resume_path,
            models=save_models,
            optimizers=save_opts,
            cumulative_step=cumulative_offset + args.total_timesteps,
        )
        print(f"[ppo] Saved resume checkpoint -> {resume_path} "
              f"({cumulative_offset + args.total_timesteps} cumulative frames).")

    np.save(
        os.path.join(results_dir, f"ppo_{variant}_episodic_returns_run_{args.run_number}.npy"),
        np.array(episodic_returns),
    )

    # Post-training evaluation: log per-agent/team return for this chunk.
    run_eval(cumulative_offset + args.total_timesteps)