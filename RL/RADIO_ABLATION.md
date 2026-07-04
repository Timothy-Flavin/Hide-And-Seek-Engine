# Radio (Communication) Ablation

Opt-in decentralized experiment where agents learn to **share observations over
the radio** via a trainable per-agent radio head. Enable per runner:

```bash
python -m RL.cleanrl_ppo --no-centralized --use-radio --level levels/test_level
python -m RL.cleanrl_dqn --no-centralized --use-radio --level levels/test_level
python -m RL.cleanrl_sac --no-centralized --use-radio --level levels/test_level
```

Results are saved as the variant `decentralized_radio` (e.g.
`experiments/results/<level>/ppo_decentralized_radio_episodic_returns_run_N.npy`)
and appear as a distinct curve in the per-level combined plot.

## What the radio does

The C++ engine's radio action per agent is a discrete index:

- `0` = no broadcast (the value every other experiment sends).
- `1 .. n_agents-1` = broadcast this agent's discovered tiles / POI knowledge to
  one specific peer (self is skipped in the enumeration).

So the radio action space has `n_agents` values (the "bins" for SAC). The
existing `centralized`, `decentralized`, and `decentralized_ego` variants keep
sending `0` and have **no radio head** — they are byte-for-byte unchanged. All
radio code is gated behind `use_radio` (which is forced off for centralized).

## Model / learning design

The radio action is treated as an **independent, additively-factored action**
alongside the movement action. This is valid because the reward is VDN-summed
across agents and the two action factors are conditionally independent given the
observation.

### Single forward pass (performance)

Network passes dominate runtime, so the radio factor must **not** add an encoder
pass or an extra head kernel. Instead of a second head module, the existing
policy/advantage head is **widened** to output `sum(discrete_dims) =
num_move_actions + num_radio_actions`. Each network does exactly one encoder pass
and one fused `nn.Sequential` head; the combined logits are then **zero-copy
split** (`logits[..., :move]` / `logits[..., move:]`) into the move and radio
factors. Per-run this is `head_out_dim = num_actions + (n_radio if use_radio)`,
so non-radio runs keep the original head width and code path unchanged.

Per algorithm:

- **PPO** — the widened actor head yields move+radio logits from one pass. The
  stored logprob is the joint `move + radio` logprob, so the existing PPO
  ratio/clipping and entropy bonus cover both factors with no other change.
- **DQN** — the widened advantage head yields move+radio advantages from one
  pass. The joint VDN Q is `V + Σ_a adv_move_a[move] + Σ_a adv_radio_a[radio]`;
  the TD target maxes over the move and radio factors independently. Radio
  actions are ε-greedy and stored in the replay buffer.
- **SAC** — the widened actor head (discrete bins) and widened advantage heads on
  both Q-networks yield both factors per pass. The soft target Q, the
  taken-action Q, the actor objective, and the autotuned-α entropy target all
  include the radio factor additively.

Every network therefore performs a **single forward pass** whether or not radio
is enabled; the only added work is the wider final projection and a couple of
zero-copy tensor views.

## Notes

- Radio requires decentralized mode; `--use-radio --centralized` raises an error.
- `--ego-view` and `--use-radio` compose. In `run_experiments.sh` the radio
  config is `decentralized_ego_radio` (ego-centric obs + radio). Variant names are
  built compositionally (`decentralized[_ego][_radio]`), so radio info-sharing is
  written into the full-map accumulation and then cropped to each agent's ego
  window.
- `run_experiments.sh` trains four configs per algorithm/level: `centralized`,
  `decentralized`, `decentralized_ego`, `decentralized_ego_radio`.
