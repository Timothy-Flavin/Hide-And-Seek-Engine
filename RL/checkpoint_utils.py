"""Shared helpers for the RL runners: result-path layout and periodic weight
checkpoints.

Results layout (per level, per model):
    experiments/results/<level>/<alg>/<alg>_<variant>_episodic_returns_run_<seed>.npy
    experiments/results/<level>/<alg>/checkpoints/<alg>_<variant>_run_<seed>_pctNNN.pt

Weights are saved at 20/40/60/80/100% of training (5 checkpoints per run/seed).
"""
import os
import torch


def variant_name(centralized: bool, ego_view: bool, use_radio: bool) -> str:
    """Compositional experiment-config name, e.g. decentralized_ego_radio."""
    if centralized:
        return "centralized"
    name = "decentralized"
    if ego_view:
        name += "_ego"
    if use_radio:
        name += "_radio"
    return name


def results_dir_for(level: str, alg: str) -> str:
    """experiments/results/<level_name>/<alg>/ (created)."""
    level_name = os.path.basename(os.path.normpath(level))
    d = os.path.join("experiments/results", level_name, alg)
    os.makedirs(d, exist_ok=True)
    return d


def _unwrap(module):
    """Return the underlying nn.Module behind a torch.compile wrapper."""
    return getattr(module, "_orig_mod", module)


class CheckpointSaver:
    """Saves model weights at fixed fractions of total training.

    Call maybe_save(global_step, state_fn) each iteration; state_fn is a
    zero-arg callable returning the dict to torch.save (built lazily so it only
    runs when a checkpoint is actually due). Call flush_remaining() once after
    the loop to guarantee all fractions (in particular 100%) are written.
    """

    def __init__(self, out_dir, prefix, total_timesteps,
                 fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
        self.dir = os.path.join(out_dir, "checkpoints")
        self.prefix = prefix
        self.total = int(total_timesteps)
        # (percent, absolute-step-threshold), ascending.
        self.schedule = [(int(round(f * 100)), int(round(f * total_timesteps)))
                         for f in fractions]
        self.saved = set()
        os.makedirs(self.dir, exist_ok=True)

    def _write(self, pct, state, global_step):
        path = os.path.join(self.dir, f"{self.prefix}_pct{pct:03d}.pt")
        torch.save(state, path)
        self.saved.add(pct)
        print(f"[checkpoint] {self.prefix} pct{pct:03d} saved at step {global_step} -> {path}")

    def maybe_save(self, global_step, state_fn):
        due = [(pct, thr) for pct, thr in self.schedule
               if pct not in self.saved and global_step >= thr]
        if not due:
            return
        state = state_fn()  # build once, reuse for any co-triggered thresholds
        for pct, _ in due:
            self._write(pct, state, global_step)

    def flush_remaining(self, state_fn):
        """Write any not-yet-saved fractions (e.g. 100% when integer-division
        truncation stops global_step just short of total_timesteps)."""
        remaining = [pct for pct, _ in self.schedule if pct not in self.saved]
        if not remaining:
            return
        state = state_fn()
        for pct in remaining:
            self._write(pct, state, self.total)


def module_state(**modules):
    """Build a checkpoint dict of {name: state_dict} for the given (possibly
    torch.compile-wrapped) modules."""
    return {name: _unwrap(m).state_dict() for name, m in modules.items()}
