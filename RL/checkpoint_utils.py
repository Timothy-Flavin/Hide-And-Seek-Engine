"""Shared helpers for the RL runners: result-path layout and periodic weight
checkpoints.

Results layout (per level, per model):
    experiments/results/<level>/<alg>/<alg>_<variant>_episodic_returns_run_<seed>.npy
    experiments/results/<level>/<alg>/checkpoints/<alg>_<variant>_run_<seed>_pctNNN.pt

Weights are saved at 20/40/60/80/100% of training (5 checkpoints per run/seed).
"""
import os
import time
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


def _triton_supports(device) -> bool:
    """torch.compile's GPU backend (Triton/inductor) needs CUDA capability
    >= 7.0. Pascal and older cards (e.g. the GTX 1080 Ti, cap 6.1) are
    unsupported and crash the inductor backend. Non-CUDA devices compile via
    the C++ backend and are left alone."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return True
    major, _ = torch.cuda.get_device_capability(dev)
    return major >= 7


def maybe_compile(module, device):
    """``torch.compile(module)`` when the target device supports it, otherwise
    the eager module unchanged. This lets old GPUs (e.g. GTX 1080 Ti, CUDA
    cap 6.1) fall back to eager instead of crashing with
    ``RuntimeError: ... too old to be supported by the triton GPU compiler``.
    Honors ``TORCHDYNAMO_DISABLE`` for CPU smoke tests."""
    if os.environ.get("TORCHDYNAMO_DISABLE"):
        return module
    if not _triton_supports(device):
        return module
    return torch.compile(module)


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


# --------------------------------------------------------------------------- #
# Resumable training checkpoints (for the human-BC 100k-frame chunk loop)
# --------------------------------------------------------------------------- #
# CheckpointSaver above snapshots *weights only* at fixed fractions for
# evaluation. Resuming a partially-trained agent for another chunk additionally
# needs the optimizer state, the exploration/anneal bookkeeping (cumulative
# frames trained across chunks), and the RNG state. Those live in a single
# rolling file per run so each chunk overwrites the previous resume point.

def resume_checkpoint_path(out_dir: str, prefix: str) -> str:
    """<out_dir>/checkpoints/<prefix>_resume.pt (dir created)."""
    d = os.path.join(out_dir, "checkpoints")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{prefix}_resume.pt")


def save_resume(path, *, models: dict, optimizers: dict, cumulative_step: int,
                extra: dict | None = None):
    """Write a resume checkpoint atomically. ``models``/``optimizers`` are dicts
    of {name: module-or-optimizer}; modules may be torch.compile-wrapped."""
    import numpy as np

    state = {
        "models": {n: _unwrap(m).state_dict() for n, m in models.items()},
        "optimizers": {n: o.state_dict() for n, o in optimizers.items()},
        "cumulative_step": int(cumulative_step),
        "extra": extra or {},
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "timestamp": time.time(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_resume(path, map_location="cpu"):
    if not path or not os.path.exists(path):
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_resume(state: dict, models: dict, optimizers: dict | None = None) -> None:
    """Load weights/optimizer state from a resume checkpoint in place. Modules are
    unwrapped, so this works whether or not they are torch.compile-wrapped."""
    for name, module in models.items():
        if name in state.get("models", {}):
            _unwrap(module).load_state_dict(state["models"][name])
    if optimizers:
        for name, opt in optimizers.items():
            if name in state.get("optimizers", {}):
                opt.load_state_dict(state["optimizers"][name])


def restore_rng(state: dict) -> None:
    """Restore torch/numpy/cuda RNG. RNG states must be CPU ByteTensors; a
    checkpoint loaded onto CUDA has moved them, so bring them back."""
    import numpy as np

    if state.get("torch_rng") is not None:
        torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
    if state.get("numpy_rng") is not None:
        np.random.set_state(state["numpy_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda_rng"]])
        except Exception:
            pass


def strip_compile_prefix(state_dict: dict) -> dict:
    """Drop torch.compile's ``_orig_mod.`` prefix so weights load into an eager
    module (used by the recorder's teammate policy)."""
    p = "_orig_mod."
    if any(k.startswith(p) for k in state_dict):
        return {(k[len(p):] if k.startswith(p) else k): v for k, v in state_dict.items()}
    return state_dict
